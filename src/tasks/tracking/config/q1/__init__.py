from calendar import c
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import  MotionTrackingOnPolicyRunner

from .env_cfgs import q1_flat_tracking_env_cfg
from .rl_cfg import  q1_tracking_ppo_runner_cfg


register_mjlab_task(
  task_id="Q1-Tracking-No-State-Estimation",
  env_cfg=q1_flat_tracking_env_cfg(has_state_estimation=False, copy_motion_end=True),
  play_env_cfg=q1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=q1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
