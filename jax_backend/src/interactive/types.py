from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax.numpy as jnp


Array = jnp.ndarray


@dataclass
class StructuralConfig:
    """Configuration values that require a full reset/rebuild when changed."""

    N: int = 30
    dt: float = 0.01
    n_sectors: int = 4
    sector_angle: float = 60.0
    mode: str = "nolearning"  # "nolearning" or "learning"
    noise_horizon_seconds: float = 60.0


@dataclass
class LiveConfig:
    """Configuration values that can be applied while the simulation is running."""

    dist_thr: float = 5.0
    z_h: float = 0.01
    z_hprime: float = 0.01
    z_action: float = 0.01
    pi_z_spatial: float = 1.0
    pi_w_spatial: float = 1.0
    s_z: float = 1.0
    s_w: float = 1.0
    alpha: float = 0.5
    eta_orders: tuple[float, ...] = (1.0, 0.0, 0.0)
    infer_lr: float = 0.1
    nsteps_infer: int = 1
    action_lr: float = 0.1
    nsteps_action: int = 1
    learning_lr: float = 0.001
    nsteps_learning: int = 1
    normalize_v: bool = True
    speed: float = 1.0


@dataclass
class LearningConfig:
    """Learning-specific configuration for interactive mode."""

    active_learnables: tuple[str, ...] = ("s_z",)


@dataclass
class StepDiagnostics:
    """Scalar diagnostics at one simulation step."""

    alignment: float
    cohesion: float
    n_connected_components: int
    mean_nearest_neighbor_distance: float
    group_speed: float
    free_energy_mean: float
    free_energy_std: float
    sensory_pe_mean: float
    process_pe_mean: float


@dataclass
class SimulationSnapshot:
    """Current state + recent history for UI clients."""

    step_idx: int
    sim_time: float
    pos: Array
    vel: Array
    mu: Array
    preparams: Optional[Dict[str, Array]]
    diagnostics: StepDiagnostics
    metric_history: Dict[str, Array]
    free_energy_history: Array
    position_history: Array
    metadata: Dict[str, Any] = field(default_factory=dict)
