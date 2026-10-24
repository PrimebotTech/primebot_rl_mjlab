# PrimeBot RL Mjlab


## ✳️ Overview
PrimeBot RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/PrimebotTech/mjlab), using MuJoCo as its
physics simulation backend, currently supporting the Q1 humanoid robot.

> [!IMPORTANT]
> This project depends on our maintained fork of mjlab:
> [https://github.com/PrimebotTech/mjlab](https://github.com/PrimebotTech/mjlab).
> Please install mjlab from this repository (see [setup.md](doc/setup_en.md)).

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical robots for real-world execution.


## 🛠️ Usage Guide

### 1. Motion Imitation Training

Train a Q1 robot to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 1.1 Prepare Motion Files

Prepare csv motion files and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/q1/dance1_subject1.csv \
--output-name dance1_subject1.npz \
--input-fps 30 \
--output-fps 50
```

**npz files will be stored at:** `src/assets/motions/q1/...`

#### 1.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Q1-Tracking-No-State-Estimation --motion_file=src/assets/motions/q1/q1_getUp.npz --env.scene.num-envs=4096
```

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**: `logs/rsl_rl/<robot>_tracking/<date_time>/model_<iteration>.pt`

### 2. Simulation Validation

To visualize policy behavior in MuJoCo:

Motion imitation:
```bash
python scripts/play.py Q1-Tracking-No-State-Estimation --motion_file=src/assets/motions/q1/q1_getUp.npz --checkpoint_file=logs/rsl_rl/q1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**:

- During training, policy.onnx is also exported for deployment onto physical robots.

Available tasks:
- Tracking: `Q1-Tracking-No-State-Estimation`


## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/PrimebotTech/mjlab): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
