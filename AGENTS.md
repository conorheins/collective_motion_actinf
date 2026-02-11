# AGENTS.md (local working context)

## Repository overview
- Root contains two implementations:
- `jax_backend/`: primary Python/JAX implementation for collective active inference simulations.
- `julia/`: Julia implementation (out of scope for current cleanup pass).

## JAX backend structure
- `jax_backend/src/`: simulation, inference, learning, geometry, and demo scripts.
- `jax_backend/src/interactive/`: interactive engine + config/learnable abstractions for real-time control.
- `jax_backend/tests/`: smoke tests for imports, tiny no-learning demo execution, and with-learning save regression.
- `jax_backend/README_JAX.md`: canonical setup/run/troubleshooting documentation.
- `jax_backend/pyproject.toml` + `jax_backend/uv.lock`: dependency source of truth.

## Interactive explorer handoff notes
- Main interactive entrypoint: `jax_backend/src/demo_interactive.py`.
- Core runtime engine: `jax_backend/src/interactive/engine.py` (`InteractiveSimulationEngine`).
- Core interactive types: `jax_backend/src/interactive/types.py`.
- Learnable parameter registry and extension hooks: `jax_backend/src/interactive/learnables.py`.
- VFE decomposition helper used by diagnostics: `jax_backend/src/genmodel/vfe.py::compute_vfe_vectorized_components`.

### Interactive control semantics
- Live-updated controls (apply without reset): `z_h`, `z_hprime`, `z_action`, `pi_z_spatial`, `pi_w_spatial`, `alpha`, `eta_order_*`, `infer_lr`, `action_lr`, `learning_lr`, `speed`.
- Reset-required controls (queued until reset/apply): `N`, `n_sectors`, `sector_angle`, `dt`, and mode (`nolearning`/`learning`).
- Learning mode supports extensible built-ins via registry (`s_z`, `s_w`, `alpha`, `eta_order_k`).

### Interactive validation commands
- Headless smoke: `uv run python src/demo_interactive.py --headless-smoke --headless-steps 20`
- Full tests (includes interactive smoke/regression): `uv run pytest`

## Conventions for agents
- Run commands from `jax_backend/` unless a task requires repo root.
- Use `uv sync`/`uv run`; do not mix in ad hoc `pip install` in this project env.
- Keep simulation code JAX-vectorized and functional where possible.
- Preserve existing script entrypoints in `src/` unless migration is explicitly requested.

## Dependency and environment workflow
- Python support policy: `>=3.11`.
- CPU environment: `uv sync --group cpu --group dev --frozen`.
- CUDA environment: `uv sync --group cuda12 --group dev --frozen`.
- If environment drift occurs, re-run sync with `--frozen`.

## Validation commands
- `uv sync --group cpu --group dev --frozen`
- `uv run pytest`
- `uv run python src/demo_nolearning.py --seed 2 --N 10 --dt 0.01 --T 20 --last_T_seconds 10`

## CI entrypoint
- `.github/workflows/jax-backend-smoke.yml`
- Matrix: Python 3.11 and 3.12, CPU smoke tests only.
