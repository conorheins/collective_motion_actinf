from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import fields, replace
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

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
        self._default_live = replace(self.engine.live_config)
        self._default_structural = replace(self.engine.structural_config)

        self.is_running = True
        self.steps_per_frame = 4
        self.selected_agent = 0

        self._pending_live: Dict[str, Any] = {}
        self._last_live_change_s: float | None = None
        self._pending_structural: Dict[str, Any] = {}
        self._pending_learning: Dict[str, Any] = {}
        self._pending_seed: int | None = None

        self._mu_history: deque[np.ndarray] = deque(maxlen=self.engine.history_window)
        self._mu_step_history: deque[int] = deque(maxlen=self.engine.history_window)
        self._preparam_history: deque[dict[str, np.ndarray]] = deque(maxlen=self.engine.history_window)
        self._preparam_step_history: deque[int] = deque(maxlen=self.engine.history_window)
        self._click_perturb_scale = 1.0

        self._build_figure()
        self._initialize_artists()

        self._animation = FuncAnimation(self.fig, self._on_timer, interval=50, blit=False, cache_frame_data=False)
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_swarm_click)

    def show(self) -> None:
        plt.show()

    def _initialize_artists(self) -> None:
        """Initialize all panels once before the timer loop starts."""
        self._refresh_plots()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(18, 11))

        self.ax_swarm = self.fig.add_axes([0.05, 0.42, 0.45, 0.53])
        self.ax_belief_orders = [
            self.fig.add_axes([0.52, 0.79, 0.22, 0.16]),
            self.fig.add_axes([0.52, 0.61, 0.22, 0.16]),
            self.fig.add_axes([0.52, 0.43, 0.22, 0.16]),
        ]
        self.ax_vfe = self.fig.add_axes([0.76, 0.79, 0.22, 0.16])
        self.ax_errors = self.fig.add_axes([0.76, 0.61, 0.22, 0.16])
        self.ax_learning = self.fig.add_axes([0.76, 0.43, 0.22, 0.16])
        self.ax_metrics = self.fig.add_axes([0.785, 0.205, 0.195, 0.205])
        self.ax_metrics.axis("off")

        self.status_text = self.ax_metrics.text(0.0, 1.0, "", va="top", ha="left", fontsize=9, linespacing=1.1)

        # Buttons (single horizontal row)
        btn_y = 0.355
        self.ax_btn_play = self.fig.add_axes([0.05, btn_y, 0.082, 0.04])
        self.ax_btn_step = self.fig.add_axes([0.14, btn_y, 0.072, 0.04])
        self.ax_btn_reset = self.fig.add_axes([0.22, btn_y, 0.105, 0.04])
        self.ax_btn_defaults = self.fig.add_axes([0.333, btn_y, 0.125, 0.04])
        self.ax_btn_seed = self.fig.add_axes([0.466, btn_y, 0.105, 0.04])
        self.ax_btn_prev = self.fig.add_axes([0.579, btn_y, 0.092, 0.04])
        self.ax_btn_next = self.fig.add_axes([0.679, btn_y, 0.092, 0.04])

        self.btn_play = Button(self.ax_btn_play, "Pause")
        self.btn_step = Button(self.ax_btn_step, "Step")
        self.btn_reset = Button(self.ax_btn_reset, "Reset/Apply")
        self.btn_defaults = Button(self.ax_btn_defaults, "Knob Defaults")
        self.btn_seed = Button(self.ax_btn_seed, "Random Seed")
        self.btn_prev = Button(self.ax_btn_prev, "Prev Agent")
        self.btn_next = Button(self.ax_btn_next, "Next Agent")

        self.btn_play.on_clicked(self._on_play_pause)
        self.btn_step.on_clicked(self._on_step_once)
        self.btn_reset.on_clicked(self._on_apply_and_reset)
        self.btn_defaults.on_clicked(self._on_reset_knobs_to_defaults)
        self.btn_seed.on_clicked(self._on_random_seed)
        self.btn_prev.on_clicked(self._on_prev_agent)
        self.btn_next.on_clicked(self._on_next_agent)

        # Learning mode toggle
        self.ax_mode_toggle = self.fig.add_axes([0.76, 0.155, 0.22, 0.04])
        self.btn_mode_toggle = Button(self.ax_mode_toggle, "")
        self.btn_mode_toggle.on_clicked(self._on_mode_toggle_click)
        self._refresh_mode_toggle_button()

        # Learnable toggles
        ndo_x = self.engine.base_genmodel.get("ndo_x", 3)
        labels = self.engine.learnable_registry.list_available(ndo_x)
        self.ax_learnables = self.fig.add_axes([0.76, 0.01, 0.22, 0.13])
        current_active = set(self.engine.learning_config.active_learnables)
        states = [label in current_active for label in labels]
        self.learnables_check = CheckButtons(self.ax_learnables, labels, states)
        self.learnables_check.on_clicked(self._on_learnables_change)
        self.ax_learnables.set_title("Learnable Params")

        # Sliders
        self._sliders: dict[str, Slider] = {}
        self._add_sliders()
        self._eta_boxes: dict[int, TextBox] = {}
        self._add_eta_text_boxes()

        self.fig.suptitle("Interactive Collective Motion Explorer", fontsize=16)

    def _add_sliders(self) -> None:
        live = self.engine.live_config
        struct = self.engine.structural_config

        slider_specs = [
            ("z_h", "z_h", 0.0001, 0.25, live.z_h, False, "live"),
            ("z_hprime", "z_hprime", 0.0001, 0.25, live.z_hprime, False, "live"),
            ("z_action", "z_action", 0.0001, 0.25, live.z_action, False, "live"),
            ("pi_z_spatial", "pi_z_spatial", 0.05, 5.0, live.pi_z_spatial, False, "live"),
            ("pi_w_spatial", "pi_w_spatial", 0.05, 5.0, live.pi_w_spatial, False, "live"),
            ("alpha", "alpha", 0.01, 2.0, live.alpha, False, "live"),
            ("infer_lr", "infer_lr", 0.001, 1.0, live.infer_lr, False, "live"),
            ("action_lr", "action_lr", 0.001, 1.0, live.action_lr, False, "live"),
            ("learning_lr", "learning_lr", 0.00001, 0.05, live.learning_lr, False, "live"),
            ("speed", "speed", 0.1, 3.0, live.speed, False, "live"),
            ("N", "N", 4, 200, struct.N, True, "struct"),
            ("n_sectors", "n_sectors", 2, 10, struct.n_sectors, True, "struct_even"),
            ("sector_angle", "sector_angle", 20.0, 170.0, struct.sector_angle, False, "struct"),
            ("dt", "dt", 0.005, 0.05, struct.dt, False, "struct"),
            ("history", "history", 50, 2000, self.engine.history_window, True, "history"),
        ]

        base_y = 0.31
        row_h = 0.03
        slider_w = 0.24
        col_positions = [0.05, 0.44]

        for idx, (key, label, vmin, vmax, init, is_int, kind) in enumerate(slider_specs):
            row = idx // 2
            col = idx % 2
            ax = self.fig.add_axes([col_positions[col], base_y - row * row_h, slider_w, 0.018])
            valstep = 1 if is_int else None
            slider = Slider(ax, label, vmin, vmax, valinit=init, valstep=valstep)
            slider.label.set_fontsize(8)
            slider.valtext.set_fontsize(8)
            slider.on_changed(self._make_slider_handler(key, kind, is_int))
            self._sliders[key] = slider

    def _add_eta_text_boxes(self) -> None:
        eta_values = [self.engine.live_config.eta_orders[idx] if idx < len(self.engine.live_config.eta_orders) else 0.0 for idx in range(3)]

        box_y = 0.038
        box_w = 0.12
        box_h = 0.03
        for order_idx in range(3):
            ax = self.fig.add_axes([0.05 + (order_idx * 0.15), box_y, box_w, box_h])
            box = TextBox(ax, f"eta_order_{order_idx}", initial=f"{eta_values[order_idx]:.3f}")
            box.on_submit(self._make_eta_submit_handler(order_idx))
            self._eta_boxes[order_idx] = box

    def _make_eta_submit_handler(self, order_idx: int):
        def _handler(text_value: str) -> None:
            try:
                value = float(text_value.strip())
            except ValueError:
                self._eta_boxes[order_idx].ax.set_facecolor("#f8d7da")
                self.fig.canvas.draw_idle()
                return

            self._eta_boxes[order_idx].ax.set_facecolor("white")
            eta_orders = list(self.engine.live_config.eta_orders)
            while len(eta_orders) <= order_idx:
                eta_orders.append(0.0)
            eta_orders[order_idx] = value
            self._pending_live["eta_orders"] = tuple(eta_orders)
            self._last_live_change_s = time.time()

        return _handler

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
                self._mu_step_history = deque(
                    list(self._mu_step_history)[-self.engine.history_window :], maxlen=self.engine.history_window
                )
                self._preparam_history = deque(
                    list(self._preparam_history)[-self.engine.history_window :], maxlen=self.engine.history_window
                )
                self._preparam_step_history = deque(
                    list(self._preparam_step_history)[-self.engine.history_window :], maxlen=self.engine.history_window
                )
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
        selected_learnables = self._get_selected_learnables_from_widget()
        learning_updates = {"active_learnables": selected_learnables}
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
        self._mu_step_history = deque(maxlen=self.engine.history_window)
        self._preparam_history = deque(maxlen=self.engine.history_window)
        self._preparam_step_history = deque(maxlen=self.engine.history_window)
        self._refresh_mode_toggle_button()
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

    def _effective_mode(self) -> str:
        return str(self.engine.structural_config.mode)

    def _refresh_mode_toggle_button(self) -> None:
        mode = self._effective_mode()
        if mode == "learning":
            color = "#2e7d32"
            hovercolor = "#388e3c"
            label = "LEARNING ON (click to disable)"
        else:
            color = "#c62828"
            hovercolor = "#d32f2f"
            label = "LEARNING OFF (click to enable)"

        self.ax_mode_toggle.set_facecolor(color)
        self.btn_mode_toggle.color = color
        self.btn_mode_toggle.hovercolor = hovercolor
        self.btn_mode_toggle.label.set_text(label)
        self.btn_mode_toggle.label.set_color("white")
        self.btn_mode_toggle.label.set_fontsize(9)

    def _on_mode_toggle_click(self, _event: Any) -> None:
        current_mode = self._effective_mode()
        new_mode = "learning" if current_mode == "nolearning" else "nolearning"
        learning_updates = {"active_learnables": self._get_selected_learnables_from_widget()}
        self.engine.apply_structural_and_reset(structural_updates={"mode": new_mode}, learning_updates=learning_updates)
        self._pending_structural.pop("mode", None)
        self.selected_agent = min(self.selected_agent, self.engine.structural_config.N - 1)
        self._mu_history = deque(maxlen=self.engine.history_window)
        self._mu_step_history = deque(maxlen=self.engine.history_window)
        self._preparam_history = deque(maxlen=self.engine.history_window)
        self._preparam_step_history = deque(maxlen=self.engine.history_window)
        self._refresh_mode_toggle_button()
        self._refresh_plots()

    def _on_reset_knobs_to_defaults(self, _event: Any) -> None:
        live_field_names = {f.name for f in fields(LiveConfig)}
        struct_field_names = {f.name for f in fields(StructuralConfig)}

        for key, slider in self._sliders.items():
            if key in live_field_names:
                slider.set_val(getattr(self._default_live, key))
            elif key in struct_field_names:
                slider.set_val(getattr(self._default_structural, key))
            elif key == "history":
                slider.set_val(400)

        eta_defaults = list(self._default_live.eta_orders[:3])
        for idx, value in enumerate(eta_defaults):
            if idx in self._eta_boxes:
                self._eta_boxes[idx].set_val(f"{value:.3f}")

        self._pending_live["eta_orders"] = tuple(eta_defaults)
        self._last_live_change_s = time.time()

    def _on_learnables_change(self, _label: str) -> None:
        active = self._get_selected_learnables_from_widget()
        self._pending_learning["active_learnables"] = active

    def _get_selected_learnables_from_widget(self) -> tuple[str, ...]:
        labels = list(self.learnables_check.labels)
        states = self.learnables_check.get_status()
        active = tuple(label.get_text() for label, state in zip(labels, states) if state)
        return active

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
        self._refresh_mode_toggle_button()
        pos = np.asarray(snap.pos)
        vel = np.asarray(snap.vel)

        if (not self._mu_step_history) or (snap.step_idx != self._mu_step_history[-1]):
            self._mu_history.append(np.asarray(snap.mu))
            self._mu_step_history.append(int(snap.step_idx))
            if snap.preparams is not None:
                self._preparam_history.append({k: np.asarray(v) for k, v in snap.preparams.items()})
                self._preparam_step_history.append(int(snap.step_idx))

        self.ax_swarm.cla()
        trail = np.asarray(snap.position_history)
        self._draw_history_traces(trail, int(np.clip(self.selected_agent, 0, pos.shape[0] - 1)))
        self.ax_swarm.scatter(pos[:, 0], pos[:, 1], s=25, color="tab:blue", alpha=0.9)
        self.ax_swarm.quiver(
            pos[:, 0],
            pos[:, 1],
            vel[:, 0],
            vel[:, 1],
            angles="xy",
            scale_units="xy",
            scale=24.0,
            color="0.45",
            alpha=0.60,
            width=0.0014,
            headwidth=2.4,
            headlength=3.0,
            headaxislength=2.8,
        )

        agent = int(np.clip(self.selected_agent, 0, pos.shape[0] - 1))
        self.ax_swarm.scatter([pos[agent, 0]], [pos[agent, 1]], s=75, color="tab:red", label=f"Agent {agent}")

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

        self._refresh_belief_panels(agent)
        self._refresh_vfe_panel(snap, agent)
        self._refresh_error_panel(snap)
        self._refresh_learning_panel(snap, agent)
        self._refresh_metric_text(snap)

        self.fig.canvas.draw_idle()

    def _draw_history_traces(self, position_history: np.ndarray, selected_agent: int) -> None:
        if position_history.shape[0] < 2:
            return

        trail_len = min(24, position_history.shape[0])
        history = position_history[-trail_len:]
        n_segments = history.shape[0] - 1
        n_agents = history.shape[1]

        all_segments = np.stack([history[:-1], history[1:]], axis=2).reshape(-1, 2, 2)
        base_colors = plt.cm.Greys(np.linspace(0.45, 0.85, n_segments))
        base_colors[:, 3] = np.linspace(0.12, 0.55, n_segments)
        all_colors = np.repeat(base_colors, n_agents, axis=0)
        lc_all = LineCollection(all_segments, colors=all_colors, linewidths=2.2, zorder=1)
        self.ax_swarm.add_collection(lc_all)

        selected_history = history[:, selected_agent, :]
        selected_segments = np.stack([selected_history[:-1], selected_history[1:]], axis=1)
        selected_colors = plt.cm.Reds(np.linspace(0.45, 0.9, n_segments))
        selected_colors[:, 3] = np.linspace(0.45, 0.98, n_segments)
        lc_selected = LineCollection(selected_segments, colors=selected_colors, linewidths=3.6, zorder=2)
        self.ax_swarm.add_collection(lc_selected)

    def _refresh_belief_panels(self, agent: int) -> None:
        for idx, axis in enumerate(self.ax_belief_orders):
            axis.cla()

        if not self._mu_history:
            for idx, axis in enumerate(self.ax_belief_orders):
                axis.set_title(f"Beliefs Order {idx}")
            return

        mu_hist = np.stack(list(self._mu_history), axis=0)
        t_axis = np.asarray(self._mu_step_history)
        mu_agent = mu_hist[:, :, agent]

        ndo_x = self.engine.base_genmodel["ndo_x"]
        ns_x = self.engine.base_genmodel["ns_x"]
        mu_agent = mu_agent.reshape(mu_agent.shape[0], ndo_x, ns_x)
        cmap = plt.cm.tab10(np.linspace(0.0, 1.0, ns_x))

        for order_idx, axis in enumerate(self.ax_belief_orders):
            if order_idx >= ndo_x:
                axis.set_title(f"Beliefs Order {order_idx}")
                axis.text(0.5, 0.5, "N/A", ha="center", va="center", transform=axis.transAxes)
                axis.set_xticks([])
                axis.set_yticks([])
                continue

            for sector_idx in range(ns_x):
                axis.plot(
                    t_axis,
                    mu_agent[:, order_idx, sector_idx],
                    color=cmap[sector_idx],
                    linewidth=1.2,
                    label=f"s{sector_idx}",
                )
            axis.set_title(f"Beliefs Order {order_idx}")
            axis.set_ylabel("$\\mu$")
            if order_idx == (len(self.ax_belief_orders) - 1):
                axis.set_xlabel("Simulation Step")
            if order_idx == 0:
                axis.legend(loc="upper right", fontsize=6, ncol=2)

    def _refresh_vfe_panel(self, snap, agent: int) -> None:
        self.ax_vfe.cla()
        hist = snap.metric_history

        fe_history = np.asarray(snap.free_energy_history)
        if fe_history.shape[0] == 0:
            self.ax_vfe.set_title("VFE")
            return

        t_axis = np.arange(fe_history.shape[0])
        fe_agent = fe_history[:, agent]
        fe_mean = np.asarray(hist["free_energy_mean"])

        self.ax_vfe.plot(t_axis, fe_agent, color="tab:red", linewidth=1.9, label=f"agent {agent}")
        self.ax_vfe.plot(t_axis, fe_mean, color="tab:blue", linewidth=1.2, alpha=0.65, label="group mean")
        self.ax_vfe.set_title("Variational Free Energy")
        self.ax_vfe.set_xlabel("Recent Steps")
        self.ax_vfe.legend(loc="upper right", fontsize=8)

    def _refresh_error_panel(self, snap) -> None:
        self.ax_errors.cla()
        hist = snap.metric_history

        if len(hist["sensory_pe_mean"]) == 0:
            self.ax_errors.set_title("Prediction Error Terms")
            return

        t_axis = np.arange(len(hist["sensory_pe_mean"]))
        spe = np.asarray(hist["sensory_pe_mean"])
        ppe = np.asarray(hist["process_pe_mean"])

        self.ax_errors.plot(t_axis, spe, color="tab:orange", label="sensory term")
        self.ax_errors.plot(t_axis, ppe, color="tab:green", label="process term")
        self.ax_errors.set_title("Prediction Error Terms")
        self.ax_errors.set_xlabel("Recent Steps")
        self.ax_errors.legend(loc="upper right", fontsize=8)

    def _refresh_learning_panel(self, snap, agent: int) -> None:
        self.ax_learning.cla()

        if snap.metadata["mode"] != "learning":
            self.ax_learning.set_title("Learned Params")
            self.ax_learning.text(0.5, 0.5, "Learning disabled", ha="center", va="center", transform=self.ax_learning.transAxes)
            self.ax_learning.set_xticks([])
            self.ax_learning.set_yticks([])
            return

        active = tuple(snap.metadata.get("active_learnables", ()))
        if not active or not self._preparam_history:
            self.ax_learning.set_title("Learned Params")
            self.ax_learning.text(0.5, 0.5, "No parameter history yet", ha="center", va="center", transform=self.ax_learning.transAxes)
            self.ax_learning.set_xticks([])
            self.ax_learning.set_yticks([])
            return

        t_axis = np.asarray(self._preparam_step_history)
        cmap = plt.cm.Dark2(np.linspace(0.0, 1.0, max(1, len(active))))

        for idx, name in enumerate(active):
            series_raw = [entry.get(name) for entry in self._preparam_history]
            if not series_raw or any(val is None for val in series_raw):
                continue
            series = np.asarray(series_raw)
            color = cmap[idx]
            if series.ndim == 1:
                self.ax_learning.plot(t_axis, series, color=color, linewidth=1.6, label=name)
            else:
                mean_values = series.mean(axis=1)
                selected_values = series[:, agent]
                self.ax_learning.plot(t_axis, mean_values, color=color, linewidth=1.8, label=f"{name} mean")
                self.ax_learning.plot(t_axis, selected_values, color=color, linewidth=1.0, linestyle="--", alpha=0.7, label=f"{name} agent")

        self.ax_learning.set_title("Learned Parameter Values")
        self.ax_learning.set_xlabel("Simulation Step")
        self.ax_learning.legend(loc="upper right", fontsize=7, ncol=1)

    def _refresh_metric_text(self, snap) -> None:
        d = snap.diagnostics

        pending_structural_txt = "none" if not self._pending_structural else ", ".join(sorted(self._pending_structural.keys()))
        pending_live_txt = "none" if not self._pending_live else ", ".join(sorted(self._pending_live.keys()))
        perturb_txt = getattr(self, "_perturbation_note", "none")
        self._perturbation_note = "none"

        text = (
            f"step: {snap.step_idx}\n"
            f"time: {snap.sim_time:.2f}s\n"
            f"mode: {snap.metadata['mode']}\n"
            f"selected agent: {self.selected_agent}\n"
            f"alignment: {d.alignment:.3f}\n"
            f"cohesion: {d.cohesion:.1f}\n"
            f"components: {d.n_connected_components}\n"
            f"mean nn-dist: {d.mean_nearest_neighbor_distance:.3f}\n"
            f"group speed: {d.group_speed:.3f}\n"
            f"F mean: {d.free_energy_mean:.3f}\n"
            f"F std: {d.free_energy_std:.3f}\n"
            f"sensory term: {d.sensory_pe_mean:.3f}\n"
            f"process term: {d.process_pe_mean:.3f}\n"
            f"perturbation: {perturb_txt}\n"
            f"pending live: {pending_live_txt}\n"
            f"pending structural: {pending_structural_txt}"
        )
        self.status_text.set_text(text)

    def _on_swarm_click(self, event: Any) -> None:
        if event.inaxes != self.ax_swarm:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        snap = self.engine.get_snapshot()
        pos = np.asarray(snap.pos)
        if pos.size == 0:
            return

        click_xy = np.array([event.xdata, event.ydata], dtype=float)
        sq_dists = np.sum((pos - click_xy) ** 2, axis=1)
        target_agent = int(np.argmin(sq_dists))

        selected_pos = pos[target_agent]
        kick_dir = selected_pos - click_xy
        kick_norm = float(np.linalg.norm(kick_dir))
        if kick_norm <= 1e-12:
            vel = np.asarray(snap.vel[target_agent])
            vel_norm = float(np.linalg.norm(vel))
            if vel_norm <= 1e-12:
                kick_dir = np.array([1.0, 0.0], dtype=float)
            else:
                kick_dir = np.array([-vel[1], vel[0]], dtype=float)
            kick_norm = float(np.linalg.norm(kick_dir))

        kick_unit = kick_dir / kick_norm
        base_kick = max(0.25, 0.5 * float(self.engine.live_config.speed))
        kick = tuple((self._click_perturb_scale * base_kick * kick_unit).tolist())

        self.engine.apply_agent_perturbation(target_agent, mode="velocity", velocity_delta=kick)
        self.selected_agent = target_agent
        self._perturbation_note = f"agent {target_agent}"
        self._refresh_plots()


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
    parser.add_argument("--z_action", type=float, default=0.001)
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
