import argparse  # 解析命令行参数用于配置运行  
import time  # 提供计时与延时功能  

import mujoco,mujoco_viewer  # MuJoCo 物理引擎接口  
import numpy as np  # 数值计算与数组处理  
import onnxruntime  # ONNX 运行时推理引擎  
import onnx  # ONNX 模型解析加载  
import torch  # PyTorch 张量与模型工具  
from scipy.spatial.transform import Rotation as R
from pynput import keyboard
from tqdm import tqdm
from loop_rate_limiters import RateLimiter

XML_PATH = "src/assets/robots/q1_v3/xml/q1_v3_mc.xml" # MuJoCo 模型 XML 路径
SIMULATION_DURATION = 300.0  # 总仿真时长（秒）  
SIMULATION_DT = 0.002  # 物理仿真步长  
CONTROL_DECIMATION = 10  # 控制更新降采样倍数  
NUM_ACTIONS = 22  # 动作维度数量  
NUM_OBS = 119  # 观测维度数量  
MOTION_ANCHOR_INDEX = 0

JOINT_XML = [  # MuJoCo 关节顺序列表  
    "left_hip_pitch_joint",  # 左髋俯仰关节名称  
    "left_hip_roll_joint",  # 左髋滚转关节名称  
    "left_hip_yaw_joint",  # 左髋偏航关节名称  
    "left_knee_joint",  # 左膝关节名称  
    "left_ankle_pitch_joint",  # 左踝俯仰关节名称  
    "left_ankle_roll_joint",  # 左踝滚转关节名称  
    "right_hip_pitch_joint",  # 右髋俯仰关节名称  
    "right_hip_roll_joint",  # 右髋滚转关节名称  
    "right_hip_yaw_joint",  # 右髋偏航关节名称  
    "right_knee_joint",  # 右膝关节名称  
    "right_ankle_pitch_joint",  # 右踝俯仰关节名称  
    "right_ankle_roll_joint",  # 右踝滚转关节名称  
    "waist_roll_joint",  # 腰部滚转关节名称  
    "waist_yaw_joint",  # 腰部偏航关节名称  
    "left_shoulder_pitch_joint",  # 左肩俯仰关节名称  
    "left_shoulder_roll_joint",  # 左肩滚转关节名称  
    "left_shoulder_yaw_joint",  # 左肩偏航关节名称  
    "left_elbow_joint",  # 左肘关节名称  
    "right_shoulder_pitch_joint",  # 右肩俯仰关节名称  
    "right_shoulder_roll_joint",  # 右肩滚转关节名称  
    "right_shoulder_yaw_joint",  # 右肩偏航关节名称  
    "right_elbow_joint",  # 右肘关节名称  
]  # 关节顺序列表结束  

class cmd:
    camera_follow = False
    reset_requested = False
    start_action = False

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")
    
    @classmethod
    def reset(cls):
        print(f"Reset")

def on_press(key):
    try:
        if key.char == 'f':
            cmd.toggle_camera_follow()
        elif key.char == '0':
            cmd.reset_requested = True
        elif key.char == 's':
            cmd.start_action = True
    except AttributeError:
        pass

def on_release():
    pass

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_apply_inverse(quat, vec):
    """Apply inverse (conjugate) quaternion rotation to a vector.

    Equivalent to rotating *vec* by the conjugate of *quat*.
    quat : (4,) or (B, 4)  wxyz
    vec  : (3,) or (B, 3)
    Returns rotated vector with same shape as *vec*.
    """
    quat = np.asarray(quat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    q_w = quat[..., 0]
    q_x = quat[..., 1]
    q_y = quat[..., 2]
    q_z = quat[..., 3]
    q_vec = np.stack([q_x, q_y, q_z], axis=-1)
    t = 2.0 * np.cross(q_vec, vec)
    return vec - q_w[..., None] * t + np.cross(q_vec, t)

def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):  # 计算 A 到 B 的相对位姿  
    rotm_a = np.zeros(9)  # 初始化旋转矩阵扁平数组  
    mujoco.mju_quat2Mat(rotm_a, quat_a)  # 四元数转旋转矩阵  
    rotm_a = rotm_a.reshape(3, 3)  # 重塑为 3x3 矩阵  
    rel_pos = rotm_a.T @ (pos_b - pos_a)  # 计算相对位置向量  
    rel_quat = quat_multiply(quat_conjugate(quat_a), quat_b)  # 计算相对旋转四元数  
    rel_quat = rel_quat / np.linalg.norm(rel_quat)  # 归一化四元数  
    return rel_pos, rel_quat  # 返回相对位置和相对姿态  

def quat_conjugate(q):  # 计算四元数共轭  
    return np.array([q[0], -q[1], -q[2], -q[3]])  # 标量不变向量取负  

def quat_multiply(q1, q2):  # 计算四元数乘积  
    w1, x1, y1, z1 = q1  # 解包第一个四元数  
    w2, x2, y2, z2 = q2  # 解包第二个四元数  
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # 计算结果 w 分量  
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # 计算结果 x 分量  
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # 计算结果 y 分量  
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # 计算结果 z 分量  
    
    return np.array([w, x, y, z])  # 返回乘积四元数  
def normalize_quaternion(quat, eps=1e-12):
    norm = np.linalg.norm(quat)
    if norm < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm

def quat_to_rot6d(quat):
    quat = normalize_quaternion(quat).astype(np.float32)
    w, x, y, z = quat
    rot = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)
    return rot[:, :2].reshape(-1)

def pd_control(target_q, q, kp, target_dq, dq, kd):  # 计算 PD 控制力矩  
    return (target_q - q) * kp + (target_dq - dq) * kd  # 位置与速度误差加权求和  

def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = np.array([data.sensor('jointpos_left_hip_pitch_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_hip_roll_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_hip_yaw_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_knee_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_ankle_pitch_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_ankle_roll_joint').data.astype(np.float32),                
                  
                  data.sensor('jointpos_right_hip_pitch_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_hip_roll_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_hip_yaw_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_knee_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_ankle_pitch_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_ankle_roll_joint').data.astype(np.float32),
                  
                  data.sensor('jointpos_waist_roll_joint').data.astype(np.float32),
                  data.sensor('jointpos_waist_yaw_joint').data.astype(np.float32),
                  
                  data.sensor('jointpos_left_shoulder_pitch_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_shoulder_roll_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_shoulder_yaw_joint').data.astype(np.float32),
                  data.sensor('jointpos_left_elbow_joint').data.astype(np.float32),

                  data.sensor('jointpos_right_shoulder_pitch_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_shoulder_roll_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_shoulder_yaw_joint').data.astype(np.float32),
                  data.sensor('jointpos_right_elbow_joint').data.astype(np.float32),
                  ])
    
    dq = np.array([data.sensor('jointvel_left_hip_pitch_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_hip_roll_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_hip_yaw_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_knee_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_ankle_pitch_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_ankle_roll_joint').data.astype(np.float32),               
                  
                  data.sensor('jointvel_right_hip_pitch_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_hip_roll_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_hip_yaw_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_knee_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_ankle_pitch_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_ankle_roll_joint').data.astype(np.float32),
                  
                  data.sensor('jointvel_waist_roll_joint').data.astype(np.float32),
                  data.sensor('jointvel_waist_yaw_joint').data.astype(np.float32),
                  
                  data.sensor('jointvel_left_shoulder_pitch_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_shoulder_roll_joint').data.astype(np.float32),
                  data.sensor('jointvel_left_shoulder_yaw_joint').data.astype(np.float32),  
                  data.sensor('jointvel_left_elbow_joint').data.astype(np.float32),
                  
                  data.sensor('jointvel_right_shoulder_pitch_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_shoulder_roll_joint').data.astype(np.float32),
                  data.sensor('jointvel_right_shoulder_yaw_joint').data.astype(np.float32),  
                  data.sensor('jointvel_right_elbow_joint').data.astype(np.float32),
                  ])
    
    q = np.squeeze(q)
    dq = np.squeeze(dq)
    quat = data.sensor("body-orientation").data[[1, 2, 3 ,0]].astype(np.float32)
    
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.float32)
    omega = data.sensor("body-angular-velocity").data.astype(np.float32)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float32)
    state_tau = data.qfrc_actuator.astype(np.float32) - data.qfrc_bias.astype(np.float32)

    return (q, dq, quat, v, omega, gvec, state_tau)
def parse_args():  # 解析命令行参数  
    parser = argparse.ArgumentParser()  # 创建参数解析器  
    parser.add_argument("--motion_file", type=str, default="./motion.npz", help="motion npz file")  # 动作文件路径参数  
    parser.add_argument("--policy_path", type=str, default="./policy.onnx", help="onnx policy")  # 策略模型路径参数  
    return parser.parse_args()  # 返回解析后的参数  


def load_policy_metadata(policy_path, joint_xml):  # 读取策略元数据并重排关节  
    model = onnx.load(policy_path)  # 加载 ONNX 模型  
    joint_seq = []  # 初始化关节名称序列  
    joint_pos_array_seq = None  # 初始化默认关节角度序列  
    joint_pos_array = None  # 初始化重排后默认角度  
    stiffness_array = None  # 初始化重排后刚度  
    damping_array = None  # 初始化重排后阻尼  
    action_scale = None  # 初始化动作缩放  
    anchor_body_name = None  # 初始化锚点刚体名称  
    body_names = None  # 初始化刚体名称列表  
    for prop in model.metadata_props:  # 遍历模型元数据属性  
        values = prop.value.split(",")  # 解析属性字符串为列表  
        if prop.key == "joint_names":  # 处理关节名称字段  
            joint_seq = values  # 保存关节名称顺序  
        elif prop.key == "default_joint_pos":  # 处理默认关节角度  
            joint_pos_array_seq = np.array([float(x) for x in values])  # 转为浮点数组  
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "joint_stiffness":  # 处理关节刚度  
            stiffness_array = np.array([float(x) for x in values])  # 转为浮点数组  
            stiffness_array = np.array([stiffness_array[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "joint_damping":  # 处理关节阻尼  
            damping_array = np.array([float(x) for x in values])  # 转为浮点数组  
            damping_array = np.array([damping_array[joint_seq.index(joint)] for joint in joint_xml])  # 按 MuJoCo 顺序重排  
        elif prop.key == "action_scale":  # 处理动作缩放  
            action_scale = np.array([float(x) for x in values])  # 转为浮点数组  
        elif prop.key == "anchor_body_name":  # 处理锚点刚体名称  
            anchor_body_name = prop.value  # 记录锚点刚体名称  
        elif prop.key == "body_names":  # 处理刚体名称列表  
            body_names = values  # 记录刚体名称列表  
        print(f"{prop.key}: {prop.value}")  # 打印元数据方便检查  

    if not joint_seq:  # 检查关节名称是否缺失  
        raise ValueError("ONNX 元数据缺少 joint_names，无法对齐关节顺序")  # 抛出缺失错误  
    if joint_pos_array_seq is None or joint_pos_array is None:  # 检查默认关节角度是否缺失  
        raise ValueError("ONNX 元数据缺少 default_joint_pos，无法初始化关节角度")  # 抛出缺失错误  
    if stiffness_array is None or damping_array is None or action_scale is None:  # 检查控制参数是否缺失  
        raise ValueError("ONNX 元数据缺少关节参数或动作缩放，请检查 joint_stiffness/joint_damping/action_scale")  # 抛出缺失错误  

    return (  # 返回整理后的策略参数  
        joint_seq,  # 关节名称顺序  
        joint_pos_array_seq,  # 默认关节角度原序列  
        joint_pos_array,  # 默认关节角度 MuJoCo 顺序  
        stiffness_array,  # 关节刚度 MuJoCo 顺序  
        damping_array,  # 关节阻尼 MuJoCo 顺序  
        action_scale,  # 动作缩放系数  
        anchor_body_name,  # 锚点刚体名称  
        body_names,  # 刚体名称列表  
    )  # 返回结束  


def load_motion(motion_file):  # 读取动作数据文件  
    motion = np.load(motion_file)  # 加载 npz 数据  
    motion_pos = motion["body_pos_w"]  # 读取所有刚体位置序列  
    motion_quat = motion["body_quat_w"]  # 读取所有刚体姿态四元数序列  
    motion_input_pos = motion["joint_pos"]  # 读取关节位置序列  
    motion_input_vel = motion["joint_vel"]  # 读取关节速度序列  
    return motion_pos, motion_quat, motion_input_pos, motion_input_vel  # 返回动作数据  


if __name__ == "__main__":  # 主程序入口  
    loop = True  # 是否循环播放动作
    print("=" * 60)
    print(f"Loop playback: {loop}")
    print("Keyboard control instructions:")
    print("  0 key: Reset all speeds to 0")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()
    
    # args = parse_args()  # 解析命令行参数  
    motion_file = "src/assets/motions/q1/q1_getUp.npz"
    policy_path = "logs/rsl_rl/q1_tracking/2026-10-10_17-46-17/2026-10-10_17-46-17.onnx"

    motion_pos, motion_quat, motion_input_pos, motion_input_vel = load_motion(motion_file)  # 加载动作数据  
    num_frames = min(motion_input_pos.shape[0], motion_input_vel.shape[0], motion_pos.shape[0], motion_quat.shape[0])
    (  # 解包策略元数据  
        joint_seq,  # 关节名称顺序  
        joint_pos_array_seq,  # 默认关节角度原序列  
        joint_pos_array,  # 默认关节角度 MuJoCo 顺序  
        stiffness_array,  # 关节刚度 MuJoCo 顺序  
        damping_array,  # 关节阻尼 MuJoCo 顺序  
        action_scale,  # 动作缩放系数  
        anchor_body_name,  # 锚点刚体名称  
        body_names,  # 刚体名称列表  
    ) = load_policy_metadata(policy_path, JOINT_XML)  # 加载策略元数据
    print(f"Loaded policy metadata with {len(joint_seq)} joints and body names: {body_names}")  # 打印加载信息

    obs = np.zeros(NUM_OBS, dtype=np.float32)  # 初始化观测向量  
    counter = 0  # 控制器计数器  

    m = mujoco.MjModel.from_xml_path(XML_PATH)  # 加载 MuJoCo 模型  
    m.opt.timestep = SIMULATION_DT
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d = mujoco.MjData(m)  # 创建 MuJoCo 数据对象  

    # 使用ONNX Runtime库创建了一个推理会话，用于加载和执行预训练的神经网络策略模型
    # policy_path = "logs/rsl_rl/q1_tracking/2026-03-28_10-34-02/policy.onnx"
    policy = onnxruntime.InferenceSession(policy_path)  

    action_buffer = np.zeros((NUM_ACTIONS,), dtype=np.float32)  # 初始化上一动作缓存  
    timestep = 0  # 初始化动作序列索引  
    target_dof_pos = joint_pos_array  # 目标关节位置初始化为默认角度  
    # d.qpos[2] = 0.42  # 设置初始基座高度  
    d.qpos[:3] = motion_pos[timestep,MOTION_ANCHOR_INDEX,:]  # 设置初始基座位置和motion file 第0帧对齐  
    d.qpos[3:7] = motion_quat[timestep,MOTION_ANCHOR_INDEX,:]  # 设置初始基座旋转和motion file 第0帧对齐  
    d.qpos[7:] = motion_input_pos[timestep,:]  # 写入 初始关节角度  
    # d.qpos[7:] = target_dof_pos  # 写入初始关节角度  
    mujoco.mj_step(m, d)  # 执行一步仿真以应用初始状态

    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()

    body_name = anchor_body_name  
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)  # 获取锚点 ID  
    print(f"ancher body name:{body_name},body id :{body_id}")
    if body_id == -1:  # 检查锚点是否存在  
        raise ValueError(f"Body {body_name} not found in model")  # 抛出错误提示  
    mode = 'window'
    viewer = mujoco_viewer.MujocoViewer(m, d, mode=mode, width=1920, height=1080)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 1]

    control_freq = 1.0 / (SIMULATION_DT * CONTROL_DECIMATION)
    start = time.time()  # 记录仿真起始时间  
    for step in tqdm(range(int(SIMULATION_DURATION / SIMULATION_DT)), desc="Simulating..."):
        step_start = time.time()  # 记录本步起始时间  
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            d.qpos[:] = initial_qpos
            d.qpos[7:] = joint_pos_array
            d.qvel[:] = initial_qvel
            cmd.reset()
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            cmd.reset_requested = False
            cmd.start_action = False
            timestep = 0
        # if loop and num_frames > 0:
        #     timestep = timestep % num_frames
        if timestep >= num_frames:
            timestep = num_frames - 1
            
        q, dq, quat, v, omega, gvec, state_tau = get_obs(d)
        q = q[-NUM_ACTIONS:]
        dq = dq[-NUM_ACTIONS:]

        if counter % CONTROL_DECIMATION == 0:  # 到达控制周期更新策略  
            position = d.xpos[body_id]  # 获取仿真中锚点刚体位置  
            quaternion = d.xquat[body_id]  # 获取仿真中锚点刚体姿态  
            motion_input = np.concatenate(  # 拼接目标关节位置与速度  
                (motion_input_pos[timestep, :], motion_input_vel[timestep, :]),  # 使用 npz 原始关节顺序  
                axis=0,  # 进行拼接  
            )  # 拼接结束  

            ref_anchor_quat = motion_quat[timestep, 0, :]
            # q_rel = quat_multiply(quat_conjugate(quaternion), ref_anchor_quat)
            # 计算ref_anchor_quat的重力向量映射
            ref_gvec = quat_apply_inverse(ref_anchor_quat, np.array([0, 0, -1]))

            offset = 0  # 观测向量写入偏移量  
            obs[offset : offset + 44] = motion_input  # 写入目标关节指令  
            # print(f"motion_input:{motion_input}")
            offset += 44  # 更新偏移量  
            # obs[offset : offset + 6] = quat_to_rot6d(q_rel)  # 参考锚点姿态
            obs[offset : offset + 3] = ref_gvec  # 参考锚点姿态
            offset += 3  # 更新偏移量  
            obs[offset : offset + 3] = gvec  # 写入重力向量映射
            # print(f"gvec:{gvec}")
            offset += 3  # 更新偏移量  
            obs[offset : offset + 3] = d.qvel[3:6]  # 写入基座角速度  
            # print(f"d.qvel[3:6]:{d.qvel[3:6]}")
            offset += 3  # 更新偏移量  
            qpos_xml = d.qpos[7 : 7 + NUM_ACTIONS]  # 读取 MuJoCo 顺序关节角度  
            obs[offset : offset + NUM_ACTIONS] = qpos_xml - joint_pos_array_seq  # 写入关节角度偏差  
            # print(f"qpos_diff:{qpos_xml - joint_pos_array_seq}")
            offset += NUM_ACTIONS  # 更新偏移量  
            qvel_xml = d.qvel[6 : 6 + NUM_ACTIONS]  # 读取 MuJoCo 顺序关节速度  
            obs[offset : offset + NUM_ACTIONS] = qvel_xml  # 写入关节速度  
            # print(f"qvel_xml:{qvel_xml}")
            offset += NUM_ACTIONS  # 更新偏移量  
            obs[offset : offset + NUM_ACTIONS] = action_buffer  # 写入上一动作  
            # print(f"action_buffer:{action_buffer}")

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # 转为批量张量输入  
            action = policy.run(  # 执行 ONNX 推理  
                ["actions"],  # 指定输出节点名称  
                {"obs": obs_tensor.numpy(), "time_step": np.array([timestep], dtype=np.float32).reshape(1, 1)},  # 传入观测与时间步  
            )[0]  # 获取推理输出  
            # action = policy.run(['actions'], {
            #     'obs': obs_tensor.numpy(),
            # })[0]
            action = np.asarray(action).reshape(-1)  # 转为一维数组  
            action_buffer = action.copy()  # 更新上一动作缓存  
            target_dof_pos = action * action_scale + joint_pos_array_seq   # 计算目标关节角度  
            target_dof_pos = target_dof_pos.reshape(-1,)  # 确保为一维向量  
            viewer.render()
            timestep += 1  # 更新动作序列索引  
            rate_limiter = RateLimiter(frequency=control_freq, warn=False)
            rate_limiter.sleep()


        tau = pd_control(  # 计算关节控制力矩  
        target_dof_pos,  # 目标关节角度  
        d.qpos[7:],  # freejoint 后面的所有关节位置  
        stiffness_array,  # 关节刚度  
        np.zeros_like(damping_array),  # 目标关节速度置零  
        d.qvel[6:],  # freejoint 后面的所有关节角速度  
        damping_array,  # 关节阻尼  
        )  # 控制力矩计算结束  
        tau = np.clip(tau, -18, 18)  # Clamp torques
        d.ctrl[:] = tau  # 写入力矩控制输入  
        
        mujoco.mj_step(m, d)
        counter += 1  # 更新控制器计数  
        time_until_next_step = m.opt.timestep - (time.time() - step_start)  # 计算剩余睡眠时间  
        if time_until_next_step > 0:  # 若剩余时间为正则休眠  
            time.sleep(time_until_next_step)  # 休眠保持实时步长  
    viewer.close()
    keyboard_listener.stop()