from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

import jax.numpy as jnp
from jax import vmap

from genmodel.defaults import parameterize_A0_no_coupling

from .types import LiveConfig


Array = jnp.ndarray
GenModel = Dict[str, Any]


@dataclass(frozen=True)
class LearnableSpec:
    """Specification for a learnable parameter family."""

    name: str
    init_fn: Callable[[int, LiveConfig], Array]
    apply_batch: Callable[[GenModel, Array, LiveConfig], GenModel]
    description: str = ""


class LearnableRegistry:
    """Registry of supported learnable parameters plus extension hooks."""

    def __init__(self) -> None:
        self._specs: Dict[str, LearnableSpec] = {}
        self._factories: Dict[str, Callable[[str], LearnableSpec]] = {}
        self._register_builtin_specs()

    def register_spec(self, spec: LearnableSpec) -> None:
        self._specs[spec.name] = spec

    def register_factory(self, prefix: str, factory: Callable[[str], LearnableSpec]) -> None:
        self._factories[prefix] = factory

    def resolve(self, name: str, ndo_x: int) -> LearnableSpec:
        if name in self._specs:
            return self._specs[name]

        if name.startswith("eta_order_"):
            order_idx = int(name.split("eta_order_")[1])
            if order_idx < 0 or order_idx >= ndo_x:
                raise ValueError(f"eta order {order_idx} outside [0, {ndo_x - 1}]")
            return _make_eta_order_spec(order_idx)

        for prefix, factory in self._factories.items():
            if name.startswith(prefix):
                return factory(name)

        raise KeyError(f"Unknown learnable parameter '{name}'")

    def list_available(self, ndo_x: int) -> tuple[str, ...]:
        base = list(self._specs.keys())
        base.extend([f"eta_order_{idx}" for idx in range(ndo_x)])
        return tuple(base)

    def _register_builtin_specs(self) -> None:
        self.register_spec(
            LearnableSpec(
                name="s_z",
                init_fn=lambda n, live: jnp.full((n,), live.s_z),
                apply_batch=_apply_s_z,
                description="Sensory smoothness (maps to Pi_z temporal precision).",
            )
        )
        self.register_spec(
            LearnableSpec(
                name="s_w",
                init_fn=lambda n, live: jnp.full((n,), live.s_w),
                apply_batch=_apply_s_w,
                description="Process smoothness (maps to Pi_w temporal precision).",
            )
        )
        self.register_spec(
            LearnableSpec(
                name="alpha",
                init_fn=lambda n, live: jnp.full((n,), live.alpha),
                apply_batch=_apply_alpha,
                description="Decay coefficient in linear flow matrices.",
            )
        )


def initialize_preparams(
    *,
    active_learnables: Iterable[str],
    registry: LearnableRegistry,
    n_agents: int,
    ndo_x: int,
    live_config: LiveConfig,
) -> Dict[str, Array]:
    out: Dict[str, Array] = {}
    for name in active_learnables:
        spec = registry.resolve(name, ndo_x)
        out[name] = spec.init_fn(n_agents, live_config)
    return out


def apply_learnables(
    *,
    genmodel: GenModel,
    preparams: Dict[str, Array],
    active_learnables: Iterable[str],
    registry: LearnableRegistry,
    ndo_x: int,
    live_config: LiveConfig,
) -> GenModel:
    updated = genmodel
    for name in active_learnables:
        if name not in preparams:
            continue
        spec = registry.resolve(name, ndo_x)
        updated = spec.apply_batch(updated, preparams[name], live_config)
    return updated


def sync_preparams_from_live(
    *,
    preparams: Dict[str, Array],
    active_learnables: Iterable[str],
    registry: LearnableRegistry,
    ndo_x: int,
    live_config: LiveConfig,
) -> Dict[str, Array]:
    if not preparams:
        return preparams

    synced: Dict[str, Array] = dict(preparams)
    for name in active_learnables:
        if name not in synced:
            continue
        n_agents = int(synced[name].shape[0])
        spec = registry.resolve(name, ndo_x)
        synced[name] = spec.init_fn(n_agents, live_config)
    return synced


def _copy_genmodel(genmodel: GenModel) -> GenModel:
    copied: GenModel = dict(genmodel)
    copied["f_params"] = dict(genmodel["f_params"])
    copied["g_params"] = dict(genmodel["g_params"])
    return copied


def _apply_s_z(genmodel: GenModel, values: Array, live_config: LiveConfig) -> GenModel:
    updated = _copy_genmodel(genmodel)
    ns_phi = int(genmodel["ns_phi"])
    spatial = live_config.pi_z_spatial * jnp.eye(ns_phi)

    def _parameterize_single(s_z: Array) -> Array:
        # This backend uses ndo_phi=2 in current demos/configs.
        temporal = jnp.diag(jnp.array([1.0, 2.0 * (s_z**2)]))
        return jnp.kron(temporal, spatial)

    updated["Pi_z"] = vmap(_parameterize_single)(values)
    return updated


def _apply_s_w(genmodel: GenModel, values: Array, live_config: LiveConfig) -> GenModel:
    updated = _copy_genmodel(genmodel)
    ns_x = int(genmodel["ns_x"])
    spatial = live_config.pi_w_spatial * jnp.eye(ns_x)

    def _parameterize_single(s_w: Array) -> Array:
        # This backend uses ndo_x=3 in current demos/configs.
        temporal = (
            jnp.diag(jnp.array([1.5, 2.0 * (s_w**2), 2.0 * (s_w**4)]))
            + jnp.diag(jnp.array([s_w**2]), k=2)
            + jnp.diag(jnp.array([s_w**2]), k=-2)
        )
        return jnp.kron(temporal, spatial)

    updated["Pi_w"] = vmap(_parameterize_single)(values)
    return updated


def _apply_alpha(genmodel: GenModel, values: Array, live_config: LiveConfig) -> GenModel:
    del live_config
    updated = _copy_genmodel(genmodel)
    a0_all = vmap(parameterize_A0_no_coupling, (0, None))(values, genmodel["ns_x"])
    updated["f_params"]["tilde_A"] = jnp.stack(genmodel["ndo_x"] * [a0_all], axis=1)
    return updated


def _make_eta_order_spec(order_idx: int) -> LearnableSpec:
    def _init(n_agents: int, live_config: LiveConfig) -> Array:
        if order_idx < len(live_config.eta_orders):
            init_value = live_config.eta_orders[order_idx]
        else:
            init_value = 0.0
        return jnp.full((n_agents,), init_value)

    def _apply(genmodel: GenModel, values: Array, live_config: LiveConfig) -> GenModel:
        del live_config
        updated = _copy_genmodel(genmodel)
        tilde_eta = updated["f_params"]["tilde_eta"]
        tilde_eta = tilde_eta.at[:, order_idx, :].set(values[:, None])
        updated["f_params"]["tilde_eta"] = tilde_eta
        return updated

    return LearnableSpec(
        name=f"eta_order_{order_idx}",
        init_fn=_init,
        apply_batch=_apply,
        description=f"Order-{order_idx} flow fixed-point belief.",
    )
