"""RL configuration for Q1 tracking task."""
from dataclasses import dataclass, field
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def q1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Q1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="q1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
  )


# ---------------------------------------------------------------------------
# DWAQ (DreamWaQ) Configuration — VAE-based velocity estimation
# ---------------------------------------------------------------------------


@dataclass
class DWAQModelCfg(RslRlModelCfg):
  """Model config for DWAQModel (VAE encoder + actor).

  This config adds DWAQ-specific parameters (history_length, cenet_out_dim,
  latent_dim, vel_dim, vel_gt_offset) on top of the standard RslRlModelCfg
  fields. When the runner converts the config to a dict via ``asdict()``,
  all fields are included and passed as ``**kwargs`` to the resolved model
  class constructor.
  """

  class_name: str = "DWAQModel"

  # ---- VAE encoder parameters ----
  history_length: int = 5
  """Number of observation steps to keep in history for VAE encoder."""

  cenet_out_dim: int = 35
  """VAE encoder output dim: 3 (vel) + 16 (μ_z) + 16 (log σ_z)."""

  latent_dim: int = 16
  """Dimension of the latent terrain code z_t."""

  vel_dim: int = 3
  """Dimension of the velocity estimate v̂_t."""

  vel_gt_offset: int | None = None
  """Start index of base_lin_vel in the concatenated critic observation.

  If None, the offset is computed dynamically in construct_algorithm()
  by reverse-calculating from the end of the critic obs:
    vel_gt_offset = critic_obs_dim - vel_dim - 3 - 3 * num_actions
  This assumes the trailing terms after base_lin_vel in critic obs are:
    base_ang_vel (3) + joint_pos (num_actions) + joint_vel (num_actions)
    + actions (num_actions)
  """


@dataclass
class DWAQPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Extended PPO algorithm config with DWAQ VAE loss matrix.

  The ``dwaq_cfg`` dict carries the VAE loss coefficients that are
  used during PPO_VAE ``update()`` to weight the encoder losses:

  - **vel_est_loss_coef**  (L_est)  : velocity estimation MSE
  - **kl_latent_loss_coef** (L_kld) : KL-divergence of latent z

  Total VAE loss = vel_est_loss_coef * L_est
                  + kl_latent_loss_coef * L_kld
  """

  class_name: str = "PPO_VAE"
  """Algorithm class name resolved by rsl_rl."""

  # ---- DWAQ VAE loss matrix ----
  dwaq_cfg: dict = field(
    default_factory=lambda: {
      "vel_est_loss_coef": 1.0,       # L_est:  velocity estimation loss weight
      "kl_latent_loss_coef": 1e-4,    # L_kld:  KL divergence for latent z (aligned with DreamWaQ reference)
      "adaboot_enabled": True,         # AdaBoot: adaptive bootstrapping (DreamWaQ §4.3)
    }
  )


@dataclass
class DWAQOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for DWAQ (DreamWaQ) with VAE velocity estimation.

  Uses ``DWAQModel`` as the actor model (which bundles encoder
  and actor MLP into a single module) and ``MLPModel`` as the critic.
  The algorithm is ``PPO_VAE`` which extends PPO with VAE losses.
  The runner is ``DWAQOnPolicyRunner`` which adds AdaBoot support.
  """

  actor: DWAQModelCfg = field(default_factory=DWAQModelCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    )
  )
  algorithm: DWAQPpoAlgorithmCfg = field(default_factory=DWAQPpoAlgorithmCfg)

  # DWAQ obs_groups: actor uses policy obs, critic uses privileged obs
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("actor",),
      "critic": ("critic",),
    }
  )


def q1_tracking_dwaq_ppo_runner_cfg() -> DWAQOnPolicyRunnerCfg:
  """Create DWAQ runner config for Q1 tracking task with VAE velocity estimation.

  The DWAQ policy uses a VAE encoder (CENet) that takes a history of
  proprioceptive observations and estimates:
    - Base velocity  v̂_t  (3-dim)
    - Latent terrain code z_t (16-dim via reparameterisation)

  Two VAE losses are added to the PPO surrogate loss during ``update()``:
    L_est  = MSE(v̂_t, v_gt)               weighted by vel_est_loss_coef
    L_kld  = KL(N(μ_z, σ_z) || N(0,1))    weighted by kl_latent_loss_coef
  """
  return DWAQOnPolicyRunnerCfg(
    actor=DWAQModelCfg(
      class_name="DWAQModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      history_length=5,
      cenet_out_dim=35,
      latent_dim=16,
      vel_dim=3,
      vel_gt_offset=None,  # Auto-computed in construct_algorithm()
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=DWAQPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      dwaq_cfg={
        "vel_est_loss_coef": 1.0,
        "kl_latent_loss_coef": 1e-4,
        "adaboot_enabled": False,
      },
    ),
    experiment_name="q1_tracking_vae",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30001,
    logger="tensorboard",
  )