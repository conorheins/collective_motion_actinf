from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import replace
from typing import Any, Dict, Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from interactive import (
    InteractiveSimulationEngine,
    LearningConfig,
    LiveConfig,
    StructuralConfig,
)


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_learnables(raw: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts if parts else ("s_z",)


class InteractiveExplorerUI:
    def __init__(self, engine: InteractiveSimulationEngine) -> None:
        self.engine = engine

        self.is_running = True
        self.steps_per_frame = 4
        self.selected_agent = 0

        self._pending_live: Dict[str, Any] = {}
        self._last_live_change_s: float | None = None
        self._pending_structural: Dict[str, Any] = {}
        self._pending_learning: Dict[str, Any] = {}
        self._pending_seed: int | None = None

        self._mu_history: deque[np.ndarray] = deque(maxlen=self.engine.history_window)

        self._build_figure()
        self._initialize_artists()

        self._animation = FuncAnimation(self.fig, self._on_timer, interval=50, blit=False, cache_frame_data=False)

    def show(self) -> None:
        plt.show()

    def _initialize_artists(self) -> None:
        """Initialize all panels once before the timer loop starts."""
        self._refresh_plots()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(18, 11))

        self.ax_swarm = self.fig.add_axes([0.05, 0.35, 0.45, 0.60])
        self.ax_beliefs = self.fig.add_axes([0.54, 0.63, 0.22, 0.32])
        self.ax_energy = self.fig.add_axes([0.54, 0.35, 0.22, 0.22])
        self.ax_metrics = self.fig.add_axes([0.78, 0.35, 0.20, 0.60])
        self.ax_metrics.axis("off")

        self.status_text = self.ax_metrics.text(0.0, 1.0, "", va="top", ha="left", fontsize=10)

        # Buttons
        btn_y = 0.29
        self.ax_btn_play = self.fig.add_axes([0.05, btn_y, 0.09, 0.04])
        self.ax_btn_step = self.fig.add_axes([0.15, btn_y, 0.09, 0.04])
        self.ax_btn_reset = self.fig.add_axes([0.25, btn_y, 0.12, 0.04])
        self.ax_btn_seed = self.fig.add_axes([0.38, btn_y, 0.12, 0.04])
        self.ax_btn_prev = self.fig.add_axes([0.54, btn_y, 0.09, 0.04])
        self.ax_btn_next = self.fig.add_axes([0.64, btn_y, 0.09, 0.04])

        self.btn_play = Button(self.ax_btn_play, "Pause")
        self.btn_step = Button(self.ax_btn_step, "Step")
        self.btn_reset = Button(self.ax_btn_reset, "Reset/Apply")
        self.btn_seed = Button(self.ax_btn_seed, "Random Seed")
        self.btn_prev = Button(self.ax_btn_prev, "Prev Agent")
        self.btn_next = Button(self.ax_btn_next, "Next Agent")

        self.btn_play.on_clicked(self._on_play_pause)
        self.btn_step.on_clicked(self._on_step_once)
        self.btn_reset.on_clicked(self._on_apply_and_reset)
        self.btn_seed.on_clicked(self._on_random_seed)
        self.btn_prev.on_clicked(self._on_prev_agent)
        self.btn_next.on_clicked(self._on_next_agent)

        # Mode toggle
        self.ax_mode_radio = self.fig.add_axes([0.78, 0.26, 0.18, 0.08])
        self.mode_radio = RadioButtons(self.ax_mode_radio, ["nolearning", "learning"])
        self.mode_radio.set_active(0 if self.engine.structural_config.mode == "nolearning" else 1)
        self.mode_radio.on_clicked(self._on_mode_change)

        # Learnable toggles
        ndo_x = self.engine.base_genmodel.get("ndo_x", 3)
        labels = self.engine.learnable_registry.list_available(ndo_x)
        self.ax_learnables = self.fig.add_axes([0.78, 0.05, 0.18, 0.20])
        current_active = set(self.engine.learning_config.active_learnables)
        states = [label in current_active for label in labels]
        self.learnables_check = CheckButtons(self.ax_learnables, labels, states)
        self.learnables_check.on_clicked(self._on_learnables_change)
        self.ax_learnables.set_title("Learnable Params")

        # Sliders
        self._sliders: dict[str, Slider] = {}
        self._add_sliders()

        self.fig.suptitle("Interactive Collective Motion Explorer", fontsize=16)

    def _add_sliders(self) -> None:
        live = self.engine.live_config
        struct = self.engine.structural_config
        eta_values = [live.eta_orders[idx] if idx < len(live.eta_orders) else 0.0 for idx in range(3)]

        slider_specs = [
            ("z_h", 0.0001, 0.25, live.z_h, False, "live"),
            ("z_hprime", 0.0001, 0.25, live.z_hprime, False, "live"),
            ("z_action", 0.0001, 0.25, live.z_action, False, "live"),
            ("pi_z_spatial", 0.05, 5.0, live.pi_z_spatial, False, "live"),
            ("pi_w_spatial", 0.05, 5.0, live.pi_w_spatial, False, "live"),
            ("alpha", 0.01, 2.0, live.alpha, False, "live"),
            ("eta_order_0", -2.0, 2.0, eta_values[0], False, "live"),
            ("eta_order_1", -2.0, 2.0, eta_values[1], False, "live"),
            ("eta_order_2", -2.0, 2.0, eta_values[2], False, "live"),
            ("infer_lr", 0.001, 1.0, live.infer_lr, False, "live"),
            ("action_lr", 0.001, 1.0, live.action_lr, False, "live"),
            ("learning_lr", 0.00001, 0.05, live.learning_lr, False, "live"),
            ("speed", 0.1, 3.0, live.speed, False, "live"),
            ("N", 4, 200, struct.N, True, "struct"),
            ("n_sectors", 2, 10, struct.n_sectors, True, "struct_even"),
            ("sector_angle", 20.0, 170.0, struct.sector_angle, False, "struct"),
            ("dt", 0.005, 0.05, struct.dt, False, "struct"),
            ("history", 50, 2000, self.engine.history_window, True, "history"),
        ]

        base_x = 0.05
        base_y = 0.22
        row_h = 0.045
        col_w = 0.22

        for idx, (name, vmin, vmax, init, is_int, kind) in enumerate(slider_specs):
            row = idx // 4
            col = idx % 4
            ax = self.fig.add_axes([base_x + col * col_w, base_y - row * row_h, 0.20, 0.02])
            valstep = 1 if is_int else None
            slider = Slider(ax, name, vmin, vmax, valinit=init, valstep=valstep)
            slider.on_changed(self._make_slider_handler(name, kind, is_int))
            self._sliders[name] = slider

    def _make_slider_handler(self, name: str, kind: str, is_int: bool):
        def _handler(value: float) -> None:
            if is_int:
                value_cast: float | int = int(round(value))
            else:
                value_cast = float(value)

            if kind == "live":
                if name.startswith("eta_order_"):
                    order_idx = int(name.split("eta_order_")[1])
                    eta_orders = list(self.engine.live_config.eta_orders)
                    while len(eta_orders) <= order_idx:
                        eta_orders.append(0.0)
                    eta_orders[order_idx] = float(value_cast)
                    self._pending_live["eta_orders"] = tuple(eta_orders)
                else:
                    self._pending_live[name] = value_cast
                self._last_live_change_s = time.time()
                return

            if kind == "history":
                self.engine.set_history_window(int(value_cast))
                self._mu_history = deque(list(self._mu_history)[-self.engine.history_window :], maxlen=self.engine.history_window)
                return

            if kind == "struct_even":
                even_value = int(value_cast)
                if even_value % 2 != 0:
                    even_value += 1
                self._pending_structural[name] = even_value
                return

            self._pending_structural[name] = value_cast

        return _handler

    def _on_play_pause(self, _event: Any) -> None:
        self.is_running = not self.is_running
        self.btn_play.label.set_text("Pause" if self.is_running else "Play")

    def _on_step_once(self, _event: Any) -> None:
        self.engine.step(1)
        self._refresh_plots()

    def _on_apply_and_reset(self, _event: Any) -> None:
        learning_updates = self._pending_learning if self._pending_learning else None
        structural_updates = self._pending_structural if self._pending_structural else None

        self.engine.apply_structural_and_reset(
            structural_updates=structural_updates,
            learning_updates=learning_updates,
            seed=self._pending_seed,
        )

        self._pending_seed = None
        self._pending_structural = {}
        self._pending_learning = {}
        self.selected_agent = min(self.selected_agent, self.engine.structural_config.N - 1)
        self._mu_history = deque(maxlen=self.engine.history_window)
        self._refresh_plots()

    def _on_random_seed(self, _event: Any) -> None:
        self._pending_seed = int(np.random.randint(0, 1_000_000))
        self._on_apply_and_reset(_event)

    def _on_prev_agent(self, _event: Any) -> None:
        self.selected_agent = (self.selected_agent - 1) % self.engine.structural_config.N
        self._refresh_plots()

    def _on_next_agent(self, _event: Any) -> None:
        self.selected_agent = (self.selected_agent + 1) % self.engine.structural_config.N
        self._refresh_plots()

    def _on_mode_change(self, label: str) -> None:
        self._pending_structural["mode"] = label

    def _on_learnables_change(self, _label: str) -> None:
        labels = list(self.learnables_check.labels)
        states = self.learnables_check.get_status()
        active = tuple(label.get_text() for label, state in zip(labels, states) if state)
        self._pending_learning["active_learnables"] = active

    def _on_timer(self, _frame: int):
        self._apply_pending_live_if_ready()

        if self.is_running:
            self.engine.step(self.steps_per_frame)

        self._refresh_plots()
        return []

    def _apply_pending_live_if_ready(self) -> None:
        if not self._pending_live:
            return

        if self._last_live_change_s is None:
            return

        if (time.time() - self._last_live_change_s) < 0.15:
            return

        self.engine.apply_live_config(**self._pending_live)
        self._pending_live = {}
        self._last_live_change_s = None

    def _refresh_plots(self) -> None:
        snap = self.engine.get_snapshot()
        pos = np.asarray(snap.pos)
        vel = np.asarray(snap.vel)

        self._mu_history.append(np.asarray(snap.mu))

        self.ax_swarm.cla()
        self.ax_swarm.scatter(pos[:, 0], pos[:, 1], s=25, color="tab:blue", alpha=0.9)
        self.ax_swarm.quiver(
            pos[:, 0],
            pos[:, 1],
            vel[:, 0],
            vel[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="0.45",
            alpha=0.65,
            width=0.002,
        )

        agent = int(np.clip(self.selected_agent, 0, pos.shape[0] - 1))
        self.ax_swarm.scatter([pos[agent, 0]], [pos[agent, 1]], s=75, color="tab:red", label=f"Agent {agent}")

        if snap.position_history.shape[0] > 1:
            trail = np.asarray(snap.position_history)
            trail_agent = trail[:, agent, :]
            self.ax_swarm.plot(trail_agent[:, 0], trail_agent[:, 1], color="tab:red", alpha=0.5, linewidth=1.3)

        x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
        y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])
        pad_x = max(0.5, 0.1 * (x_max - x_min + 1e-9))
        pad_y = max(0.5, 0.1 * (y_max - y_min + 1e-9))
        self.ax_swarm.set_xlim(x_min - pad_x, x_max + pad_x)
        self.ax_swarm.set_ylim(y_min - pad_y, y_max + pad_y)
        self.ax_swarm.set_title("2D Collective Motion")
        self.ax_swarm.set_xlabel("X")
        self.ax_swarm.set_ylabel("Y")
        self.ax_swarm.legend(loc="upper right", fontsize=8)

        self._refresh_belief_panel(agent)
        self._refresh_energy_panel(snap)
        self._refresh_metric_text(snap)

        self.fig.canvas.draw_idle()

    def _refresh_belief_panel(self, agent: int) -> None:
        self.ax_beliefs.cla()

        if not self._mu_history:
            self.ax_beliefs.set_title("Selected-Agent Beliefs")
            return

        mu_hist = np.stack(list(self._mu_history), axis=0)  # (T, n_mu, N)
        mu_agent = mu_hist[:, :, agent]

        ndo_x = self.engine.base_genmodel["ndo_x"]
        ns_x = self.engine.base_genmodel["ns_x"]
        mu_agent = mu_agent.reshape(mu_agent.shape[0], ndo_x, ns_x)
        mu_order_mean = mu_agent.mean(axis=2)

        t_axis = np.arange(mu_order_mean.shape[0])
        for order_idx in range(ndo_x):
            self.ax_beliefs.plot(t_axis, mu_order_mean[:, order_idx], label=f"order {order_idx}")

        self.ax_beliefs.set_title(f"Beliefs (Agent {agent})")
        self.ax_beliefs.set_xlabel("Recent Steps")
        self.ax_beliefs.set_ylabel("Mean $\\mu$ over sectors")
        self.ax_beliefs.legend(loc="upper right", fontsize=8)

    def _refresh_energy_panel(self, snap) -> None:
        self.ax_energy.cla()
        hist = snap.metric_history

        if len(hist["free_energy_mean"]) == 0:
            self.ax_energy.set_title("Free Energy / Error Terms")
            return

        t_axis = np.arange(len(hist["free_energy_mean"]))
        fe_mean = np.asarray(hist["free_energy_mean"])
        fe_std = np.asarray(hist["free_energy_std"])
        spe = np.asarray(hist["sensory_pe_mean"])
        ppe = np.asarray(hist["process_pe_mean"])

        self.ax_energy.plot(t_axis, fe_mean, color="tab:blue", label="F mean")
        self.ax_energy.fill_between(t_axis, fe_mean - fe_std, fe_mean + fe_std, color="tab:blue", alpha=0.2, label="F std")
        self.ax_energy.plot(t_axis, spe, color="tab:orange", label="sensory term")
        self.ax_energy.plot(t_axis, ppe, color="tab:green", label="process term")
        self.ax_energy.set_title("Free Energy + Components")
        self.ax_energy.set_xlabel("Recent Steps")
        self.ax_energy.legend(loc="upper right", fontsize=8)

    def _refresh_metric_text(self, snap) -> None:
        d = snap.diagnostics

        pending_structural_txt = "none" if not self._pending_structural else ", ".join(sorted(self._pending_structural.keys()))
        pending_live_txt = "none" if not self._pending_live else ", ".join(sorted(self._pending_live.keys()))

        text = (
            f"step: {snap.step_idx}\n"
            f"time: {snap.sim_time:.2f}s\n"
            f"mode: {snap.metadata['mode']}\n"
            f"selected agent: {self.selected_agent}\n\n"
            f"alignment: {d.alignment:.3f}\n"
            f"cohesion: {d.cohesion:.1f}\n"
            f"components: {d.n_connected_components}\n"
            f"mean nn-dist: {d.mean_nearest_neighbor_distance:.3f}\n"
            f"group speed: {d.group_speed:.3f}\n"
            f"F mean: {d.free_energy_mean:.3f}\n"
            f"F std: {d.free_energy_std:.3f}\n"
            f"sensory term: {d.sensory_pe_mean:.3f}\n"
            f"process term: {d.process_pe_mean:.3f}\n\n"
            f"pending live: {pending_live_txt}\n"
            f"pending structural: {pending_structural_txt}"
        )
        self.status_text.set_text(text)


def run_headless_smoke(args: argparse.Namespace) -> None:
    structural_config = StructuralConfig(
        N=args.N,
        dt=args.dt,
        n_sectors=args.n_sectors,
        sector_angle=args.sector_angle,
        mode=args.mode,
        noise_horizon_seconds=max(5.0, args.noise_horizon_seconds),
    )
    live_config = LiveConfig(
        z_h=args.z_h,
        z_hprime=args.z_hprime,
        z_action=args.z_action,
        pi_z_spatial=args.pi_z_spatial,
        pi_w_spatial=args.pi_w_spatial,
        alpha=args.alpha,
        eta_orders=args.eta_orders,
        infer_lr=args.infer_lr,
        action_lr=args.action_lr,
        learning_lr=args.learning_lr,
        speed=args.speed,
    )
    learning_config = LearningConfig(active_learnables=args.active_learnables)

    engine = InteractiveSimulationEngine(
        structural_config=structural_config,
        live_config=live_config,
        learning_config=learning_config,
        seed=args.seed,
        history_window=args.history_window,
    )

    engine.step(args.headless_steps)
    snap = engine.get_snapshot()
    d = snap.diagnostics
    print(
        {
            "step": snap.step_idx,
            "mode": snap.metadata["mode"],
            "alignment": round(d.alignment, 4),
            "cohesion": round(d.cohesion, 4),
            "free_energy_mean": round(d.free_energy_mean, 4),
            "n_agents": snap.metadata["n_agents"],
        }
    )


def run_ui(args: argparse.Namespace) -> None:
    structural_config = StructuralConfig(
        N=args.N,
        dt=args.dt,
        n_sectors=args.n_sectors,
        sector_angle=args.sector_angle,
        mode=args.mode,
        noise_horizon_seconds=max(30.0, args.noise_horizon_seconds),
    )
    live_config = LiveConfig(
        z_h=args.z_h,
        z_hprime=args.z_hprime,
        z_action=args.z_action,
        pi_z_spatial=args.pi_z_spatial,
        pi_w_spatial=args.pi_w_spatial,
        alpha=args.alpha,
        eta_orders=args.eta_orders,
        infer_lr=args.infer_lr,
        action_lr=args.action_lr,
        learning_lr=args.learning_lr,
        speed=args.speed,
    )
    learning_config = LearningConfig(active_learnables=args.active_learnables)

    engine = InteractiveSimulationEngine(
        structural_config=structural_config,
        live_config=live_config,
        learning_config=learning_config,
        seed=args.seed,
        history_window=args.history_window,
    )

    app = InteractiveExplorerUI(engine)
    app.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Collective Motion Explorer")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", type=str, default="nolearning", choices=["nolearning", "learning"])

    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n_sectors", type=int, default=4)
    parser.add_argument("--sector_angle", type=float, default=60.0)
    parser.add_argument("--noise_horizon_seconds", type=float, default=60.0)

    parser.add_argument("--z_h", type=float, default=0.01)
    parser.add_argument("--z_hprime", type=float, default=0.01)
    parser.add_argument("--z_action", type=float, default=0.01)
    parser.add_argument("--pi_z_spatial", type=float, default=1.0)
    parser.add_argument("--pi_w_spatial", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta_orders", type=_parse_float_tuple, default=(1.0, 0.0, 0.0))
    parser.add_argument("--infer_lr", type=float, default=0.1)
    parser.add_argument("--action_lr", type=float, default=0.1)
    parser.add_argument("--learning_lr", type=float, default=0.001)
    parser.add_argument("--speed", type=float, default=1.0)

    parser.add_argument("--active_learnables", type=_parse_learnables, default=("s_z",))
    parser.add_argument("--history_window", type=int, default=400)

    parser.add_argument("--headless-smoke", action="store_true", dest="headless_smoke", default=False)
    parser.add_argument("--headless-steps", type=int, default=20)

    args = parser.parse_args()

    if args.headless_smoke:
        run_headless_smoke(args)
    else:
        run_ui(args)
