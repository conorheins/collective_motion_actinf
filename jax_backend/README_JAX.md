# JAX backend setup instructions

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Optional for GPU installs: NVIDIA GPU, current drivers, and CUDA-compatible system setup

## Install and sync dependencies

From the repository root:

```bash
cd /Users/conorheins/Documents/collective_motion_actinf/jax_backend
```

### CPU setup

```bash
uv sync --group cpu --group dev --frozen
```

### CUDA 12 GPU setup

```bash
uv sync --group cuda12 --group dev --frozen
```

Dependency versions are controlled by `pyproject.toml` and locked in `uv.lock`.

## Running a demo

Run demo scripts through `uv run` so they use the synced environment.

```bash
uv run python src/demo_nolearning.py --seed 2 --N 10 --dt 0.01 --T 20 --last_T_seconds 10
```

You can append `--save` to persist trajectory history in a local `.npz` file.

## Interactive Explorer

A new interactive desktop explorer is available at:

```bash
uv run python src/demo_interactive.py
```

It provides:
- Real-time 2D swarm visualization
- Selected-agent belief traces split by generalized order (one panel per order, sector-wise lines)
- Free-energy and prediction-error diagnostics (including selected-agent VFE trace)
- Learning-parameter traces over time in learning mode
- Interactive knobs + controls (`Play/Pause`, `Step`, `Reset/Apply`, `Knob Defaults`, `Random Seed`, `Prev Agent`, `Next Agent`)
- Left-click perturbations on the swarm panel to give an individual a velocity kick (selected agent turns away from the click point and gets a temporary nudge)

### Headless smoke mode

Use this mode in CI/headless environments to validate startup and stepping without opening a GUI:

```bash
uv run python src/demo_interactive.py --headless-smoke --headless-steps 20
```

### Control semantics

- Live-updated controls (take effect on future steps): noise levels (`z_h`, `z_hprime`, `z_action`), model precision scales (`pi_z_spatial`, `pi_w_spatial`), flow parameters (`alpha`, `eta_order_*`), optimizer rates (`infer_lr`, `action_lr`, `learning_lr`), and `speed`.
- Reset-required controls (queued until `Reset/Apply`): `N`, `n_sectors`, `sector_angle`, and `dt`.
- Learning mode toggle is a dedicated `LEARNING ON/OFF` button (green/red) that applies immediately.
- Structural changes are intentionally reset-only so tensor shapes and compiled kernels remain consistent and performant.
- In this interactive viewer, `z_action` defaults to `0.001` (lower than the non-interactive demos).

### Observation-noise convention

- `z_h`: variance of additive observation noise on position-like (order-0 distance) observations (`h`).
- `z_hprime`: variance of additive observation noise on velocity-like observations (`h'`, first derivative of visual-distance dynamics).

## Testing

```bash
uv run pytest -q
```

## Troubleshooting

### `TypeError: 'type' object is not subscriptable` in Haiku

This usually indicates Python 3.8 with newer package versions. Use Python 3.11+ and resync:

```bash
uv sync --group cpu --group dev --python 3.11
```

### `AttributeError: module 'jax.random' has no attribute 'KeyArray'`

This comes from incompatible `jax`/`jax-md` combinations (historically seen with older `jax-md` plus newer JAX). Use the lockfile-driven install to keep compatible versions together:

```bash
uv sync --group cpu --group dev --frozen
```

### Dependency drift after manual pip installs

If `pip install ...` is run outside `uv sync`, transitive dependencies can drift and break imports. Re-sync from lockfile:

```bash
uv sync --group cpu --group dev --frozen
```

### Matplotlib figure does not show

In headless environments (CI, remote shells), use file output (`--save`) or set:

```bash
export MPLBACKEND=Agg
```
