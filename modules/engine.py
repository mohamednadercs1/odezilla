import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

class SymPyNumericalSolver:
    def __init__(self, equations, variables, x_span, y0, method="RK45", t_eval=None):
        """
        equations: list of SymPy expressions for derivatives dy_i/dx
        variables: list of SymPy functions [y1(x), y2(x), ...]
        x_span: (x_start, x_end)
        y0: list of initial values [y1_0, y2_0, ...]
        method: "RK45", "RK23", "DOP853", ...
        t_eval: optional points to evaluate
        """
        self.equations = equations
        self.variables = variables
        self.n = len(variables)
        self.x_span = x_span
        self.y0 = y0
        self.method = method
        self.t_eval = t_eval

        # Build numerical function from SymPy expressions
        self._build_numeric_function()

    def _build_numeric_function(self):
        # x symbol
        self.x_sym = list(self.variables[0].atoms(sp.Symbol))[0]

        # y symbols for lambdify
        self.y_syms = sp.symbols(f'y0:{self.n}')  # y0, y1, ...

        # Convert sympy expr → lambda functions
        self.funcs = [sp.lambdify((self.x_sym, self.y_syms), eq, "numpy")
                      for eq in self.equations]

    def _rhs(self, x_val, y_vals):
        return np.array([f(x_val, *y_vals) for f in self.funcs], dtype=float)

    def solve(self):
        sol = solve_ivp(
            fun=self._rhs,
            t_span=self.x_span,
            y0=self.y0,
            method=self.method,
            t_eval=self.t_eval
        )
        if not sol.success:
            raise ValueError(f"Solver failed: {sol.message}")
        return {"X_values": sol.t, "Y_values": sol.y}

    @staticmethod
    def get_plot_data(results_dict, title="ODE Solution"):
        series = []
        for label, data in results_dict.items():
            for i, y_vals in enumerate(data["Y_values"]):
                series.append({
                    "label": f"{label}[{i}]",
                    "x_values": list(data["X_values"]),
                    "y_values": list(y_vals)
                })
        return {
            "title": title,
            "x_label": "x",
            "y_label": "y(x)",
            "series": series
        }

    @staticmethod
    def get_plot_image(results_dict, title="ODE Solution"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, data in results_dict.items():
            for i, y_vals in enumerate(data["Y_values"]):
                ax.plot(data["X_values"], y_vals, label=f"{label}[{i}]")
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def plot_results(results_dict, title="ODE Solution"):
        plt.figure(figsize=(8, 5))
        for label, data in results_dict.items():
            for i, y_vals in enumerate(data["Y_values"]):
                plt.plot(data["X_values"], y_vals, label=f"{label}[{i}]")
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def compare(y1, y2, label1="Method1", label2="Method2"):
        y1, y2 = np.array(y1), np.array(y2)
        min_len = min(len(y1), len(y2))
        error = np.abs(y1[:min_len] - y2[:min_len])
        print(f"Max difference between {label1} and {label2}: {np.max(error)}")
        return error