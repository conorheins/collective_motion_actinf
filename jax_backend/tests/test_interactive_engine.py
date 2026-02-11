import os
import sys
from pathlib import Path

import numpy as np
from jax import random

os.environ.setdefault("MPLBACKEND", "Agg")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genmodel import compute_vfe_vectorized, compute_vfe_vectorized_components, init_genmodel
from genprocess import get_observations, init_gen_process
from interactive import InteractiveSimulationEngine, LearningConfig, LiveConfig, StructuralConfig
from utils import get_default_inits


def test_interactive_engine_nolearning_smoke() -> None:
    engine = InteractiveSimulationEngine(
        structural_config=StructuralConfig(N=8, dt=0.02, n_sectors=4, sector_angle=60.0, mode="nolearning"),
        live_config=LiveConfig(),
        seed=11,
        history_window=64,
    )

    snap = engine.step(20)

    assert snap.pos.shape == (8, 2)
    assert snap.vel.shape == (8, 2)
    assert snap.mu.shape[1] == 8
    assert np.isfinite(snap.diagnostics.free_energy_mean)
    assert np.isfinite(snap.diagnostics.alignment)


def test_interactive_engine_learning_smoke_and_preparam_updates() -> None:
    engine = InteractiveSimulationEngine(
        structural_config=StructuralConfig(N=8, dt=0.02, n_sectors=4, sector_angle=60.0, mode="learning"),
        live_config=LiveConfig(learning_lr=0.001),
        learning_config=LearningConfig(active_learnables=("s_z", "alpha", "eta_order_1")),
        seed=7,
        history_window=64,
    )

    assert engine.preparams is not None
    before = {k: np.array(v) for k, v in engine.preparams.items()}

    snap = engine.step(20)

    assert snap.preparams is not None
    changed = [not np.allclose(before[k], np.array(snap.preparams[k])) for k in before.keys()]
    assert any(changed)
    assert np.isfinite(snap.diagnostics.process_pe_mean)


def test_live_knob_update_without_reset() -> None:
    engine = InteractiveSimulationEngine(
        structural_config=StructuralConfig(N=12, dt=0.02, n_sectors=4, sector_angle=60.0, mode="nolearning"),
        seed=3,
        history_window=64,
    )

    engine.step(5)
    step_before = engine.get_snapshot().step_idx

    pre_a = np.array(engine.base_genmodel["f_params"]["tilde_A"])
    pre_noise_std = float(np.std(np.array(engine.genproc["action_noise"][engine._t_idx :])))

    engine.apply_live_config(z_action=0.08, alpha=1.2)

    post_a = np.array(engine.base_genmodel["f_params"]["tilde_A"])
    post_noise_std = float(np.std(np.array(engine.genproc["action_noise"][engine._t_idx :])))

    assert engine.get_snapshot().step_idx == step_before
    assert not np.allclose(pre_a, post_a)
    assert post_noise_std > pre_noise_std

    engine.step(10)
    assert engine.get_snapshot().step_idx == step_before + 10


def test_structural_update_requires_reset_and_rebuild() -> None:
    engine = InteractiveSimulationEngine(
        structural_config=StructuralConfig(N=10, dt=0.02, n_sectors=4, sector_angle=60.0, mode="nolearning"),
        seed=5,
    )

    engine.step(8)
    assert engine.get_snapshot().step_idx == 8

    snap = engine.apply_structural_and_reset(structural_updates={"N": 16, "sector_angle": 90.0})

    assert snap.step_idx == 0
    assert snap.pos.shape == (16, 2)
    assert engine.structural_config.N == 16
    assert np.isclose(engine.structural_config.sector_angle, 90.0)


def test_vfe_vectorized_component_decomposition_consistency() -> None:
    key = random.PRNGKey(0)
    init_dict = get_default_inits(6, 1.0, 0.02)

    pos, vel, genproc, _ = init_gen_process(key, init_dict)
    genmodel = init_genmodel(init_dict)

    mu = genmodel["f_params"]["tilde_eta"].copy().reshape(6, genmodel["ndo_x"] * genmodel["ns_x"]).T
    phi, _, empty_mask = get_observations(pos, vel, genproc, 0)

    vfe = compute_vfe_vectorized(mu, phi, empty_mask, genmodel)
    vfe_components, sensory_term, process_term = compute_vfe_vectorized_components(mu, phi, empty_mask, genmodel)

    assert np.allclose(np.array(vfe), np.array(vfe_components), atol=1e-6)
    assert np.all(np.isfinite(np.array(sensory_term)))
    assert np.all(np.isfinite(np.array(process_term)))
