"""
Module 1 – Numerical engine
Fixed-step Euler / Heun / RK4 and adaptive RK45 for dy/dt = f(t, y).
Returns T_values / Y_values (what app.py expects) and plot data with
t_values / y_values (what the Streamlit frontend expects).
"""

import io
import base64
import math

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_STEPS = 200_000
MAX_PLOT_POINTS = 1000
BLOWUP = 1e12


class SymPyNumericalSolver:
    METHODS = ("euler", "heun", "rk4", "rk45")

    def __init__(self, equation, t_span, y0, method="rk4", h=0.1):
        """
        equation : callable f(t, y) -> float   (scalar ODE)
        t_span   : (t_start, t_end)   (t_end < t_start integrates backwards)
        y0       : initial value at t_start
        method   : euler | heun | rk4 | rk45
        h        : step size for the fixed-step methods
        """
        method = (method or "rk4").lower()
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Use one of {self.METHODS}.")
        if not callable(equation):
            raise TypeError("equation must be a callable f(t, y).")
        self.t0, self.t1 = float(t_span[0]), float(t_span[1])
        if self.t0 == self.t1:
            raise ValueError("t_end must be different from t_start.")
        if not h or h <= 0:
            raise ValueError("Step size h must be > 0.")
        self.f = equation
        self.y0 = float(y0)
        self.method = method
        self.n_steps = int(min(MAX_STEPS, max(1, math.ceil(abs(self.t1 - self.t0) / h))))

    # ── one fixed step ───────────────────────────────────────────────────
    def _step(self, t, y, dt):
        f = self.f
        if self.method == "euler":
            return y + dt * f(t, y)
        if self.method == "heun":
            k1 = f(t, y)
            k2 = f(t + dt, y + dt * k1)
            return y + dt * (k1 + k2) / 2
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt * k1 / 2)
        k3 = f(t + dt / 2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self):
        warning = None
        n = self.n_steps

        if self.method == "rk45":
            t_eval = np.linspace(self.t0, self.t1, n + 1)
            with np.errstate(all="ignore"):
                sol = solve_ivp(lambda t, y: [self.f(t, y[0])],
                                (self.t0, self.t1), [self.y0], method="RK45",
                                t_eval=t_eval, rtol=1e-8, atol=1e-10)
            ts, ys = sol.t, sol.y[0]
            if not sol.success:
                if len(ts) < 2:
                    raise ValueError(f"RK45 failed: {sol.message}")
                warning = (f"RK45 stopped at t≈{ts[-1]:.4g} "
                           "(solution blows up or is singular).")
        else:
            ts = np.linspace(self.t0, self.t1, n + 1)
            ys = np.empty(n + 1)
            ys[0] = self.y0
            dt = ts[1] - ts[0]
            for i in range(n):
                try:
                    with np.errstate(all="ignore"):
                        yn = float(self._step(ts[i], ys[i], dt))
                except (ArithmeticError, ValueError, TypeError):
                    yn = float("nan")
                if not np.isfinite(yn) or abs(yn) > BLOWUP:
                    warning = (f"Solution blows up or becomes undefined near "
                               f"t≈{ts[i]:.4g}; result truncated there.")
                    ts, ys = ts[: i + 1], ys[: i + 1]
                    break
                ys[i + 1] = yn

        return {"T_values": ts, "Y_values": ys,
                "method": self.method.upper(), "warning": warning}

    # ── plotting helpers ─────────────────────────────────────────────────
    @staticmethod
    def get_plot_data(results_dict, title="ODE Solution"):
        """results_dict: {label: {"T_values": 1-D, "Y_values": 1-D or 2-D}}"""
        series = []
        for label, data in results_dict.items():
            t = np.asarray(data["T_values"], dtype=float)
            ys = np.atleast_2d(np.asarray(data["Y_values"], dtype=float))
            idx = np.arange(len(t))
            if len(t) > MAX_PLOT_POINTS:
                idx = np.unique(np.linspace(0, len(t) - 1, MAX_PLOT_POINTS).astype(int))
            for i, y in enumerate(ys):
                series.append({
                    "label": label if len(ys) == 1 else f"{label}[{i}]",
                    "t_values": t[idx].tolist(),
                    "x_values": t[idx].tolist(),      # kept for backward compatibility
                    "y_values": y[idx].tolist(),
                })
        return {"title": title, "x_label": "t", "y_label": "y(t)", "series": series}

    @staticmethod
    def get_plot_image(results_dict, title="ODE Solution"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, data in results_dict.items():
            for i, y in enumerate(np.atleast_2d(data["Y_values"])):
                ax.plot(data["T_values"], y, label=f"{label}[{i}]")
        ax.set_xlabel("t"); ax.set_ylabel("y(t)"); ax.set_title(title)
        ax.grid(True); ax.legend()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def compare(y1, y2, label1="Method1", label2="Method2"):
        y1, y2 = np.asarray(y1), np.asarray(y2)
        m = min(len(y1), len(y2))
        err = np.abs(y1[:m] - y2[:m])
        print(f"Max difference between {label1} and {label2}: {np.max(err)}")
        return err
