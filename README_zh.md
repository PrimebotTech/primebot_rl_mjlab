# PrimeBot RL Mjlab


## ✳️ 概述

PrimeBot RL Mjlab 是一个基于 [mjlab](https://github.com/PrimebotTech/mjlab) 构建的强化学习项目，
使用 MuJoCo 作为物理仿真后端，当前支持 Q1 人形机器人。

> [!IMPORTANT]
> 本项目依赖我们维护的 mjlab 分支：
> [https://github.com/PrimebotTech/mjlab](https://github.com/PrimebotTech/mjlab)。
> 请从该仓库安装 mjlab（详见 [setup.md](doc/setup_zh.md)）。

Mjlab 结合了 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的成熟高层 API 与
[MuJoCo](https://github.com/google-deepmind/mujoco_warp) 的高精度物理引擎，
为强化学习机器人研究与 Sim-to-Real（仿真到实机）部署提供了一个轻量化、模块化的框架。


## 📦 安装配置

安装和配置步骤请参考 [setup.md](doc/setup_zh.md)


## 🔁 流程概览

使用强化学习实现机器人运动控制的基本流程如下：

`训练` → `仿真验证` → `仿真到实机`

- **训练**: 在 MuJoCo 模拟环境中让机器人与环境交互，并通过奖励函数最大化学习策略。
- **仿真验证**: 加载训练好的策略进行回放，验证策略行为是否符合预期。
- **仿真到实机**: 将策略部署到物理机器人上，实现真实环境中的运动控制。


## 🛠️ 使用指南

### 1. 动作模仿训练

训练 Q1 机器人模仿参考动作序列。

<div style="margin-left: 20px;">

#### 1.1 准备动作文件

将准备好的 csv 格式的动作文件转为训练可用的 npz 文件：

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/q1/dance1_subject1.csv \
--output-name dance1_subject1.npz \
--input-fps 30 \
--output-fps 50
```

**npz文件默认保存路径为**：`src/assets/motions/q1/...`

#### 1.2 训练

确保有可用的npz文件之后，执行以下指令进行训练：

```bash
python scripts/train.py Q1-Tracking-No-State-Estimation --motion_file=src/assets/motions/q1/q1_getUp.npz --env.scene.num-envs=4096
```

</div>

> [!NOTE]
> 有关动作模仿训练的详细说明，请参阅BeyondMimic 文档
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  参数说明
- `--env.scene`: 仿真场景配置，包括环境数量（num_envs）、物理仿真步长、地面类型、重力、随机扰动等参数。
- `--env.observations`: 观测空间配置，控制训练时输入到策略网络的状态信息，如关节位置、速度、IMU等内容。
- `--env.rewards`: 奖励函数配置，定义每步训练时的优化目标。
- `--env.commands`: 控制命令配置，用于生成训练时随机或指定的动作指令。
- `--env.terminations`: 终止条件配置，定义训练 episode 的结束条件。
- `--agent.seed`: 训练随机种子，用于结果复现，不同 seed 会导致策略略有差异。
- `--agent.resume`: 是否从上次中断的 checkpoint 继续训练。 设置为 True 时，会自动加载最近一次保存的 .pt 模型文件。
- `--agent.policy`: 策略网络结构配置，例如 MLP 层数、隐藏维度、激活函数等。
- `--agent.algorithm`: 强化学习算法配置。可设置优化超参数，如学习率、批量大小、GAE λ 等。

**默认保存训练结果**：`logs/rsl_rl/<robot>_tracking/<date_time>/model_<iteration>.pt`

### 2. 仿真验证

如果想要在 MuJoCo 中查看训练效果，可以运行以下命令：

查看动作模仿训练效果：
```bash
python scripts/play.py Q1-Tracking-No-State-Estimation --motion_file=src/assets/motions/q1/q1_getUp.npz --checkpoint_file=logs/rsl_rl/q1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**说明**：

- 训练时在每次保存模型时会同步导出 policy.onnx 文件在同层目录下，可用于实物部署。

可用任务：
- 动作模仿：`Q1-Tracking-No-State-Estimation`


## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [mjlab](https://github.com/PrimebotTech/mjlab): 构建训练与运行代码的基础。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): 用于动作跟踪的通用人形机器人控制框架。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): 提供 GPU 加速渲染与仿真接口。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
