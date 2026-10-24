# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import mujoco, mujoco_viewer
import numpy as np
from tqdm import tqdm
import os
import time


COM_PROJECTION_BODY_NAME = "com_projection_marker"


def create_model_with_com_projection_marker(
    model_path,
    marker_radius=0.02,
    marker_half_height=0.004,
):
    spec = mujoco.MjSpec.from_file(model_path)
    marker_body = spec.worldbody.add_body(
        name=COM_PROJECTION_BODY_NAME,
        mocap=1,
        pos=[0.0, 0.0, marker_half_height],
    )
    marker_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[marker_radius, marker_half_height, 0.0],
        rgba=[0.0, 1.0, 0.0, 0.85],
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
    return spec.compile()


def get_robot_root_body_id(model):
    free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joint_ids) > 0:
        return int(model.jnt_bodyid[free_joint_ids[0]])
    return 1 if model.nbody > 1 else 0


def update_com_projection_marker(model, data, root_body_id, marker_mocap_id, marker_half_height):
    com_pos_w = data.subtree_com[root_body_id]
    data.mocap_pos[marker_mocap_id] = [com_pos_w[0], com_pos_w[1], marker_half_height]
    data.mocap_quat[marker_mocap_id] = [1.0, 0.0, 0.0, 0.0]


def load_motion(motion_file):  # 读取动作数据文件  
    motion = np.load(motion_file)  # 加载 npz 数据  
    motion_pos = motion["body_pos_w"]  # 读取所有刚体位置序列  
    motion_quat = motion["body_quat_w"]  # 读取所有刚体姿态四元数序列  
    motion_input_pos = motion["joint_pos"]  # 读取关节位置序列  
    motion_input_vel = motion["joint_vel"]  # 读取关节速度序列  
    return motion_pos, motion_quat, motion_input_pos, motion_input_vel  # 返回动作数据  

def run_mujoco(cfg,loop=False,motion_file=None):

    motion_pos, motion_quat, motion_input_pos, _ =load_motion(motion_file)

    num_frames = len(motion_input_pos)
    qpos_list = [np.concatenate([motion_pos[i,0,:], motion_quat[i,0,:], motion_input_pos[i]]) for i in range(num_frames)]
    print(len(qpos_list))
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    marker_half_height = 0.004
    model = create_model_with_com_projection_marker(
        cfg.sim_config.mujoco_model_path,
        marker_half_height=marker_half_height,
    )
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos=qpos_list[0]
    mujoco.mj_fwdPosition(model, data)
    root_body_id = get_robot_root_body_id(model)
    marker_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, COM_PROJECTION_BODY_NAME
    )
    marker_mocap_id = int(model.body_mocapid[marker_body_id])
    update_com_projection_marker(
        model, data, root_body_id, marker_mocap_id, marker_half_height
    )
    mujoco.mj_fwdPosition(model, data)


    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'
    
    mode = 'window'
    viewer = mujoco_viewer.MujocoViewer(model, data,width=1920, height=1080)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 1]
    viewer.render()

    count_lowlevel = 0
    start_time = time.time()
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        idx=frame_idx(count_lowlevel)
        data.qpos[:] = qpos_list[idx]
        mujoco.mj_fwdPosition(model, data)
        update_com_projection_marker(
            model, data, root_body_id, marker_mocap_id, marker_half_height
        )
        mujoco.mj_fwdPosition(model, data)
        viewer.render()

        elapsed_real_time = time.time() - start_time
        target_sim_time = (step + 1) * cfg.sim_config.dt
        if elapsed_real_time < target_sim_time:
            time.sleep(target_sim_time - elapsed_real_time)

        count_lowlevel += 1

    else:
        viewer.close()
    print("Simulation finished.")

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--motion_file',type=str,help='path to motion file(npz)')
    parser.add_argument('--loop',action="store_true",help='loop the policy')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'assets', 'robots', 'q1_v3', 'xml', 'q1_v3_mc.xml')
            sim_duration = 1000.0
            dt = 0.02
            
    run_mujoco(Sim2simCfg(),args.loop,args.motion_file)
