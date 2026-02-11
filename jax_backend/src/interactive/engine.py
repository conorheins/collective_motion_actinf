from __future__ import annotations

from collections import deque
from dataclasses import fields, replace
from typing import Any, Callable, Deque, Dict, Optional

import numpy as np
from jax import grad, jit, lax, random, tree_util, vmap
from jax import numpy as jnp

from action import infer_actions
from genmodel import compute_vfe_vectorized, compute_vfe_vectorized_components, init_genmodel
from genmodel.defaults import parameterize_A0_no_coupling
from genmodel.precisions import create_full_precision_matrix
from genprocess import advance_positions, get_observations, init_gen_process
from inference import run_inference
from utils import get_default_inits, initialize_meta_params

from .learnables import LearnableRegistry, apply_learnables, initialize_preparams
from .types import (
    LearningConfig,
    LiveConfig,
    SimulationSnapshot,
    StepDiagnostics,
    StructuralConfig,
)


class InteractiveSimulationEngine:
    """UI-agnostic simulation engine for interactive collective active inference."""

    def __init__(
        self,
        *,
        structural_config: Optional[StructuralConfig] = None,
        live_config: Optional[LiveConfig] = None,
        learning_config: Optional[LearningConfig] = None,
        seed: int = 1,
        history_window: int = 400,
        learnable_registry: Optional[LearnableRegistry] = None,
    ) -> None:
        self.structural_config = structural_config or StructuralConfig()
        self.live_config = live_config or LiveConfig()
        self.learning_config = learning_config or LearningConfig()
        self.seed = int(seed)
        self.history_window = int(history_window)
        self.learnable_registry = learnable_registry or LearnableRegistry()

        self._validate_configs()

        self.key = random.PRNGKey(self.seed)
        self._noise_key = self.key

        self.genproc: Dict[str, Any] = {}
        self.base_genmodel: Dict[str, Any] = {}
        self.meta_params: Dict[str, Any] = {}

        self.pos: jnp.ndarray = jnp.zeros((self.structural_config.N, 2))
        self.vel: jnp.ndarray = jnp.zeros((self.structural_config.N, 2))
        self.mu: jnp.ndarray = jnp.zeros((1, self.structural_config.N))
        self.preparams: Optional[Dict[str, jnp.ndarray]] = None

        self._step_fn: Optional[Callable[..., Any]] = None

        self._step_idx = 0
        self._t_idx = 0

        self._metric_history: Dict[str, Deque[float]] = {}
        self._position_history: Deque[np.ndarray]
        self._latest_diagnostics = StepDiagnostics(
            alignment=float("nan"),
            cohesion=float("nan"),
            n_connected_components=0,
            mean_nearest_neighbor_distance=float("nan"),
            group_speed=float("nan"),
            free_energy_mean=float("nan"),
            free_energy_std=float("nan"),
            sensory_pe_mean=float("nan"),
            process_pe_mean=float("nan"),
        )

        self.reset(seed=self.seed)

    def step(self, n_steps: int = 1) -> SimulationSnapshot:
        """Advance the simulation by `n_steps` and return a snapshot."""

        if n_steps < 1:
            return self.get_snapshot()

        for _ in range(int(n_steps)):
            if self._t_idx >= int(self.genproc["t_axis"].shape[0]):
                self._refresh_noise_horizon()

            if self.structural_config.mode == "learning":
                assert self.preparams is not None, "Learning mode requires preparams"
                out = self._step_fn(self.pos, self.vel, self.mu, self.preparams, self._t_idx)
                pos_next, vel_next, mu_next, preparams_next, f_vec, s_term, p_term = out
                self.preparams = preparams_next
            else:
                out = self._step_fn(self.pos, self.vel, self.mu, self._t_idx)
                pos_next, vel_next, mu_next, f_vec, s_term, p_term = out

            self.pos, self.vel, self.mu = pos_next, vel_next, mu_next
            self._step_idx += 1
            self._t_idx += 1

            self._latest_diagnostics = self._compute_python_diagnostics(f_vec=f_vec, sensory_term=s_term, process_term=p_term)
            self._append_history()

        return self.get_snapshot()

    def apply_live_config(self, **updates: Any) -> SimulationSnapshot:
        """Apply non-structural parameter updates that take effect on future steps."""

        if not updates:
            return self.get_snapshot()

        allowed_fields = {f.name for f in fields(LiveConfig)}
        unknown = set(updates.keys()) - allowed_fields
        if unknown:
            raise ValueError(f"Unknown live config fields: {sorted(unknown)}")

        previous_live = self.live_config
        self.live_config = replace(self.live_config, **updates)

        self.meta_params = initialize_meta_params(
            infer_lr=self.live_config.infer_lr,
            nsteps_infer=self.live_config.nsteps_infer,
            action_lr=self.live_config.action_lr,
            nsteps_action=self.live_config.nsteps_action,
            learning_lr=self.live_config.learning_lr,
            nsteps_learning=self.live_config.nsteps_learning,
            normalize_v=self.live_config.normalize_v,
        )

        # Keep process geometry/speed synchronized.
        self.genproc["dist_thr"] = self.live_config.dist_thr
        self.genproc["speed"] = jnp.asarray(self.live_config.speed)

        changed = {name for name in updates if getattr(previous_live, name) != getattr(self.live_config, name)}

        if {"z_h", "z_hprime", "z_action"} & changed:
            self._resample_future_noise()

        if {
            "pi_z_spatial",
            "pi_w_spatial",
            "s_z",
            "s_w",
            "alpha",
            "eta_orders",
        } & changed:
            self._apply_live_to_base_genmodel()

        if self.preparams is not None:
            n_agents = self.structural_config.N
            ndo_x = self.base_genmodel["ndo_x"]
            if "alpha" in changed and "alpha" in self.preparams:
                self.preparams["alpha"] = jnp.full((n_agents,), self.live_config.alpha)
            if "s_z" in changed and "s_z" in self.preparams:
                self.preparams["s_z"] = jnp.full((n_agents,), self.live_config.s_z)
            if "s_w" in changed and "s_w" in self.preparams:
                self.preparams["s_w"] = jnp.full((n_agents,), self.live_config.s_w)
            if "eta_orders" in changed:
                for idx in range(ndo_x):
                    key = f"eta_order_{idx}"
                    if key in self.preparams:
                        if idx < len(self.live_config.eta_orders):
                            value = self.live_config.eta_orders[idx]
                        else:
                            value = 0.0
                        self.preparams[key] = jnp.full((n_agents,), value)

        self._compile_step_fn()
        return self.get_snapshot()

    def apply_structural_and_reset(
        self,
        *,
        structural_updates: Optional[Dict[str, Any]] = None,
        learning_updates: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> SimulationSnapshot:
        """Apply structural config updates and reset simulation state."""

        if structural_updates:
            self.structural_config = replace(self.structural_config, **structural_updates)

        if learning_updates:
            self.learning_config = replace(self.learning_config, **learning_updates)

        self._validate_configs()
        return self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> SimulationSnapshot:
        """Reset with current configs, optionally changing random seed."""

        if seed is not None:
            self.seed = int(seed)

        self.key = random.PRNGKey(self.seed)

        init_dict = get_default_inits(
            self.structural_config.N,
            self.structural_config.noise_horizon_seconds,
            self.structural_config.dt,
            n_sectors=self.structural_config.n_sectors,
            sector_angle=self.structural_config.sector_angle,
        )

        eta0 = self.live_config.eta_orders[0] if self.live_config.eta_orders else 1.0
        init_dict.update(
            {
                "dist_thr": self.live_config.dist_thr,
                "z_h": self.live_config.z_h,
                "z_hprime": self.live_config.z_hprime,
                "z_action": self.live_config.z_action,
                "alpha": self.live_config.alpha,
                "eta": eta0,
                "pi_z_spatial": self.live_config.pi_z_spatial,
                "pi_w_spatial": self.live_config.pi_w_spatial,
                "s_z": self.live_config.s_z,
                "s_w": self.live_config.s_w,
            }
        )

        self.pos, self.vel, self.genproc, self._noise_key = init_gen_process(self.key, init_dict)
        self.genproc["speed"] = jnp.asarray(self.live_config.speed)

        self.base_genmodel = init_genmodel(init_dict)
        self._apply_live_to_base_genmodel()

        self.meta_params = initialize_meta_params(
            infer_lr=self.live_config.infer_lr,
            nsteps_infer=self.live_config.nsteps_infer,
            action_lr=self.live_config.action_lr,
            nsteps_action=self.live_config.nsteps_action,
            learning_lr=self.live_config.learning_lr,
            nsteps_learning=self.live_config.nsteps_learning,
            normalize_v=self.live_config.normalize_v,
        )

        n_agents = self.structural_config.N
        self.mu = self.base_genmodel["f_params"]["tilde_eta"].copy().reshape(
            n_agents,
            self.base_genmodel["ndo_x"] * self.base_genmodel["ns_x"],
        ).T

        if self.structural_config.mode == "learning":
            self.preparams = initialize_preparams(
                active_learnables=self.learning_config.active_learnables,
                registry=self.learnable_registry,
                n_agents=n_agents,
                ndo_x=self.base_genmodel["ndo_x"],
                live_config=self.live_config,
            )
        else:
            self.preparams = None

        self._step_idx = 0
        self._t_idx = 0

        self._initialize_history()
        self._latest_diagnostics = self._compute_python_diagnostics(
            f_vec=jnp.full((n_agents,), jnp.nan),
            sensory_term=jnp.full((n_agents,), jnp.nan),
            process_term=jnp.full((n_agents,), jnp.nan),
        )
        self._append_history()

        self._compile_step_fn()
        return self.get_snapshot()

    def get_snapshot(self) -> SimulationSnapshot:
        """Return a serializable snapshot for UIs/tests."""

        metric_history = {
            key: jnp.asarray(list(values), dtype=jnp.float32)
            for key, values in self._metric_history.items()
        }

        if self._position_history:
            position_history = jnp.asarray(np.stack(list(self._position_history), axis=0))
        else:
            position_history = jnp.zeros((0, self.structural_config.N, 2), dtype=jnp.float32)

        preparams = None
        if self.preparams is not None:
            preparams = {k: v for k, v in self.preparams.items()}

        return SimulationSnapshot(
            step_idx=self._step_idx,
            sim_time=self._step_idx * self.structural_config.dt,
            pos=self.pos,
            vel=self.vel,
            mu=self.mu,
            preparams=preparams,
            diagnostics=self._latest_diagnostics,
            metric_history=metric_history,
            position_history=position_history,
            metadata={
                "mode": self.structural_config.mode,
                "seed": self.seed,
                "active_learnables": self.learning_config.active_learnables,
                "n_agents": self.structural_config.N,
            },
        )

    def set_history_window(self, history_window: int) -> None:
        """Resize retained history buffers."""

        history_window = int(history_window)
        if history_window < 5:
            history_window = 5

        if history_window == self.history_window:
            return

        old_metric_history = {k: list(v) for k, v in self._metric_history.items()}
        old_positions = list(self._position_history)

        self.history_window = history_window
        self._initialize_history()

        for key, values in old_metric_history.items():
            for value in values[-self.history_window :]:
                self._metric_history[key].append(value)

        for pos in old_positions[-self.history_window :]:
            self._position_history.append(pos)

    def _initialize_history(self) -> None:
        self._metric_history = {
            "alignment": deque(maxlen=self.history_window),
            "cohesion": deque(maxlen=self.history_window),
            "components": deque(maxlen=self.history_window),
            "mean_nearest_neighbor_distance": deque(maxlen=self.history_window),
            "group_speed": deque(maxlen=self.history_window),
            "free_energy_mean": deque(maxlen=self.history_window),
            "free_energy_std": deque(maxlen=self.history_window),
            "sensory_pe_mean": deque(maxlen=self.history_window),
            "process_pe_mean": deque(maxlen=self.history_window),
        }
        self._position_history = deque(maxlen=self.history_window)

    def _append_history(self) -> None:
        d = self._latest_diagnostics
        self._metric_history["alignment"].append(d.alignment)
        self._metric_history["cohesion"].append(d.cohesion)
        self._metric_history["components"].append(float(d.n_connected_components))
        self._metric_history["mean_nearest_neighbor_distance"].append(d.mean_nearest_neighbor_distance)
        self._metric_history["group_speed"].append(d.group_speed)
        self._metric_history["free_energy_mean"].append(d.free_energy_mean)
        self._metric_history["free_energy_std"].append(d.free_energy_std)
        self._metric_history["sensory_pe_mean"].append(d.sensory_pe_mean)
        self._metric_history["process_pe_mean"].append(d.process_pe_mean)
        self._position_history.append(np.asarray(self.pos))

    def _validate_configs(self) -> None:
        if self.structural_config.mode not in {"nolearning", "learning"}:
            raise ValueError("structural_config.mode must be 'nolearning' or 'learning'")
        if self.structural_config.N < 2:
            raise ValueError("N must be >= 2")
        if self.structural_config.n_sectors < 2:
            raise ValueError("n_sectors must be >= 2")
        if self.structural_config.dt <= 0:
            raise ValueError("dt must be positive")
        if self.structural_config.noise_horizon_seconds <= self.structural_config.dt:
            raise ValueError("noise_horizon_seconds must be greater than dt")

    def _clone_genmodel(self, genmodel: Dict[str, Any]) -> Dict[str, Any]:
        cloned = dict(genmodel)
        cloned["f_params"] = dict(genmodel["f_params"])
        cloned["g_params"] = dict(genmodel["g_params"])
        return cloned

    def _apply_live_to_base_genmodel(self) -> None:
        n_agents = self.structural_config.N
        genmodel = self._clone_genmodel(self.base_genmodel)

        a0_all = vmap(parameterize_A0_no_coupling, (0, None))(
            self.live_config.alpha * jnp.ones(n_agents),
            genmodel["ns_x"],
        )
        genmodel["f_params"]["tilde_A"] = jnp.stack(genmodel["ndo_x"] * [a0_all], axis=1)

        tilde_eta = genmodel["f_params"]["tilde_eta"]
        for idx in range(genmodel["ndo_x"]):
            if idx < len(self.live_config.eta_orders):
                eta_value = self.live_config.eta_orders[idx]
            else:
                eta_value = 0.0
            tilde_eta = tilde_eta.at[:, idx, :].set(eta_value)
        genmodel["f_params"]["tilde_eta"] = tilde_eta

        genmodel["Pi_z"] = vmap(create_full_precision_matrix, (None, None, None, 0))(
            genmodel["ns_phi"],
            genmodel["ndo_phi"],
            self.live_config.pi_z_spatial,
            self.live_config.s_z * jnp.ones(n_agents),
        )
        genmodel["Pi_w"] = vmap(create_full_precision_matrix, (None, None, None, 0))(
            genmodel["ns_x"],
            genmodel["ndo_x"],
            self.live_config.pi_w_spatial,
            self.live_config.s_w * jnp.ones(n_agents),
        )

        self.base_genmodel = genmodel

    def _make_nolearning_step_fn(self) -> Callable[..., Any]:
        genproc = self.genproc
        genmodel = self.base_genmodel
        inference_params = self.meta_params["inference_params"]
        action_params = self.meta_params["action_params"]
        speed = genproc.get("speed", jnp.asarray(1.0))

        def step_fn(pos: jnp.ndarray, vel: jnp.ndarray, mu: jnp.ndarray, t_idx: int):
            phi, all_dh_dr_self, empty_sectors_mask = get_observations(pos, vel, genproc, t_idx)
            infer_res, _ = run_inference(phi, mu, empty_sectors_mask, genmodel, **inference_params)
            mu_next, epsilon_z = infer_res

            vfe, sensory_term, process_term = compute_vfe_vectorized_components(
                mu_next,
                phi,
                empty_sectors_mask,
                genmodel,
            )

            vel_next = infer_actions(
                vel,
                epsilon_z,
                genmodel,
                all_dh_dr_self,
                **action_params,
            )

            pos_next = advance_positions(
                pos,
                vel_next,
                genproc["action_noise"][t_idx],
                dt=genproc["dt"],
                speed=speed,
            )

            return pos_next, vel_next, mu_next, vfe, sensory_term, process_term

        return step_fn

    def _make_learning_step_fn(self) -> Callable[..., Any]:
        genproc = self.genproc
        base_genmodel = self.base_genmodel
        inference_params = self.meta_params["inference_params"]
        action_params = self.meta_params["action_params"]
        learning_params = self.meta_params["learning_params"]
        speed = genproc.get("speed", jnp.asarray(1.0))
        active_learnables = tuple(self.learning_config.active_learnables)
        registry = self.learnable_registry
        live_config = self.live_config
        ndo_x = base_genmodel["ndo_x"]

        def compose_genmodel(preparams: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
            return apply_learnables(
                genmodel=base_genmodel,
                preparams=preparams,
                active_learnables=active_learnables,
                registry=registry,
                ndo_x=ndo_x,
                live_config=live_config,
            )

        def loss_fn(preparams: Dict[str, jnp.ndarray], obs: jnp.ndarray, mu_prev: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            genmodel_t = compose_genmodel(preparams)
            return compute_vfe_vectorized(mu_prev, obs, mask, genmodel_t).sum()

        dfdparams = grad(loss_fn)

        learning_lr = learning_params["k_params"]
        nsteps_learning = learning_params["num_steps"]

        def step_fn(
            pos: jnp.ndarray,
            vel: jnp.ndarray,
            mu: jnp.ndarray,
            preparams: Dict[str, jnp.ndarray],
            t_idx: int,
        ):
            genmodel_t = compose_genmodel(preparams)
            phi, all_dh_dr_self, empty_sectors_mask = get_observations(pos, vel, genproc, t_idx)

            infer_res, _ = run_inference(phi, mu, empty_sectors_mask, genmodel_t, **inference_params)
            mu_next, epsilon_z = infer_res

            vfe, sensory_term, process_term = compute_vfe_vectorized_components(
                mu_next,
                phi,
                empty_sectors_mask,
                genmodel_t,
            )

            vel_next = infer_actions(
                vel,
                epsilon_z,
                genmodel_t,
                all_dh_dr_self,
                **action_params,
            )

            pos_next = advance_positions(
                pos,
                vel_next,
                genproc["action_noise"][t_idx],
                dt=genproc["dt"],
                speed=speed,
            )

            def _learn_once(current_preparams: Dict[str, jnp.ndarray], _):
                grads = dfdparams(current_preparams, phi, mu, empty_sectors_mask)
                updated = tree_util.tree_map(
                    lambda param, grad_value: param - (learning_lr * grad_value),
                    current_preparams,
                    grads,
                )
                return updated, None

            preparams_next, _ = lax.scan(_learn_once, preparams, jnp.arange(nsteps_learning))

            return pos_next, vel_next, mu_next, preparams_next, vfe, sensory_term, process_term

        return step_fn

    def _compile_step_fn(self) -> None:
        if self.structural_config.mode == "learning":
            self._step_fn = jit(self._make_learning_step_fn())
        else:
            self._step_fn = jit(self._make_nolearning_step_fn())

    def _resample_future_noise(self) -> None:
        n_total = int(self.genproc["t_axis"].shape[0])
        remaining = n_total - self._t_idx
        if remaining <= 0:
            return

        n_agents = self.structural_config.N
        ndo_phi = self.base_genmodel["ndo_phi"]
        ns_phi = self.base_genmodel["ns_phi"]

        noise_key_obs, noise_key_action, self._noise_key = random.split(self._noise_key, 3)

        z_gp = jnp.array([jnp.sqrt(self.live_config.z_h), jnp.sqrt(self.live_config.z_hprime)]).reshape(1, ndo_phi, 1, 1)
        sensory_noise = jnp.sqrt(self.structural_config.dt) * z_gp * random.normal(
            noise_key_obs,
            shape=(remaining, ndo_phi, ns_phi, n_agents),
        )

        action_noise = self.live_config.z_action * random.normal(
            noise_key_action,
            shape=(remaining, n_agents, 2),
        )

        self.genproc["sensory_noise"] = self.genproc["sensory_noise"].at[self._t_idx :].set(sensory_noise)
        self.genproc["action_noise"] = self.genproc["action_noise"].at[self._t_idx :].set(action_noise)

    def _refresh_noise_horizon(self) -> None:
        n_total = int(self.genproc["t_axis"].shape[0])
        n_agents = self.structural_config.N
        ndo_phi = self.base_genmodel["ndo_phi"]
        ns_phi = self.base_genmodel["ns_phi"]

        noise_key_obs, noise_key_action, self._noise_key = random.split(self._noise_key, 3)

        z_gp = jnp.array([jnp.sqrt(self.live_config.z_h), jnp.sqrt(self.live_config.z_hprime)]).reshape(1, ndo_phi, 1, 1)
        sensory_noise = jnp.sqrt(self.structural_config.dt) * z_gp * random.normal(
            noise_key_obs,
            shape=(n_total, ndo_phi, ns_phi, n_agents),
        )

        action_noise = self.live_config.z_action * random.normal(
            noise_key_action,
            shape=(n_total, n_agents, 2),
        )

        self.genproc["sensory_noise"] = sensory_noise
        self.genproc["action_noise"] = action_noise
        self._t_idx = 0
        self._compile_step_fn()

    def _compute_python_diagnostics(
        self,
        *,
        f_vec: jnp.ndarray,
        sensory_term: jnp.ndarray,
        process_term: jnp.ndarray,
    ) -> StepDiagnostics:
        pos = np.asarray(self.pos)
        vel = np.asarray(self.vel)

        vel_norms = np.linalg.norm(vel, axis=1, keepdims=True)
        safe_norms = np.clip(vel_norms, a_min=1e-12, a_max=None)
        vel_unit = vel / safe_norms

        alignment = float(np.linalg.norm(vel_unit.mean(axis=0)))
        group_speed = float(np.linalg.norm(vel.mean(axis=0)))

        deltas = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(deltas, axis=2)

        np.fill_diagonal(dist, np.inf)
        mean_nn = float(np.mean(np.min(dist, axis=1)))

        adjacency = dist < float(self.live_config.dist_thr)
        n_components = self._count_connected_components(adjacency)
        cohesion = 1.0 if n_components == 1 else 0.0

        f_np = np.asarray(f_vec)
        sensory_np = np.asarray(sensory_term)
        process_np = np.asarray(process_term)

        free_energy_mean = float(np.nanmean(f_np)) if not np.isnan(f_np).all() else float("nan")
        free_energy_std = float(np.nanstd(f_np)) if not np.isnan(f_np).all() else float("nan")
        sensory_mean = float(np.nanmean(sensory_np)) if not np.isnan(sensory_np).all() else float("nan")
        process_mean = float(np.nanmean(process_np)) if not np.isnan(process_np).all() else float("nan")

        return StepDiagnostics(
            alignment=alignment,
            cohesion=cohesion,
            n_connected_components=n_components,
            mean_nearest_neighbor_distance=mean_nn,
            group_speed=group_speed,
            free_energy_mean=free_energy_mean,
            free_energy_std=free_energy_std,
            sensory_pe_mean=sensory_mean,
            process_pe_mean=process_mean,
        )

    @staticmethod
    def _count_connected_components(adjacency: np.ndarray) -> int:
        n_nodes = adjacency.shape[0]
        visited = np.zeros(n_nodes, dtype=bool)
        n_components = 0

        for node in range(n_nodes):
            if visited[node]:
                continue
            n_components += 1
            stack = [node]
            visited[node] = True

            while stack:
                current = stack.pop()
                neighbours = np.where(adjacency[current])[0]
                for neighbour in neighbours:
                    if visited[neighbour]:
                        continue
                    visited[neighbour] = True
                    stack.append(neighbour)

        return n_components
