import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")          # ← no display needed (web environment)
import matplotlib.pyplot as plt
import io, base64

class NumericalSolver:
    def __init__(self, equation, t_span, y0, method="RK45", h=0.1, t_eval=None,
                 equation_string=None):
        """
        equation: function f(t, y) or None (if using equation_string)
        t_span: tuple (start, end)
        y0: initial value
        method: "euler", "rk4", or any SciPy method
        h: step size for Euler/RK4
        t_eval: array of points for SciPy
        equation_string: optional, input like "-2*y + np.sin(t)"
        """
        self.equation = equation
        self.t_span = t_span
        self.y0 = y0
        self.method = method.lower()
        self.h = h
        self.t_eval = t_eval
        self.equation_string = equation_string

        if self.equation_string:
            try:
                self.equation = lambda t, y: eval(self.equation_string, {
                    "t": t, "y": y, "np": np,
                    "cos": np.cos, "sin": np.sin, "exp": np.exp
                })
            except Exception as e:
                raise ValueError(f"Invalid equation string: {e}")

        if self.equation is None:
            raise ValueError("No equation provided!")

    # ✅ Euler Method
    def euler(self):
        t0, t_end = self.t_span
        t_values = [t0]
        y_values = [self.y0]
        t = t0
        y = self.y0
        while t < t_end:
            try:
                y = y + self.h * self.equation(t, y)
            except Exception as e:
                raise ValueError(f"Error during Euler computation at t={t}: {e}")
            t = t + self.h
            t_values.append(t)
            y_values.append(y)
        return np.array(t_values), np.array(y_values)

    # ✅ RK4 Method
    def rk4(self):
        t0, t_end = self.t_span
        t_values = [t0]
        y_values = [self.y0]
        t = t0
        y = self.y0
        h = self.h
        while t < t_end:
            try:
                f = self.equation
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h * k1 / 2)
                k3 = f(t + h / 2, y + h * k2 / 2)
                k4 = f(t + h, y + h * k3)
                y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            except Exception as e:
                raise ValueError(f"Error during RK4 computation at t={t}: {e}")
            t = t + h
            t_values.append(t)
            y_values.append(y)
        return np.array(t_values), np.array(y_values)

    # ✅ Main solve function
    def solve(self):
        method_name = self.method.lower()
        if method_name == "euler":
            t, y = self.euler()
        elif method_name == "rk4":
            t, y = self.rk4()
        else:
            try:
                solution = solve_ivp(
                    fun=self.equation,
                    t_span=self.t_span,
                    y0=[self.y0],
                    method=self.method.upper(),
                    t_eval=self.t_eval
                )
                t, y = solution.t, solution.y[0]
            except Exception as e:
                raise ValueError(f"SciPy solver error: {e}")
        return {
            "T_values": t,
            "Y_values": y
        }

    # ✅ FIX 1 – get_plot_data(): returns chart data as JSON lists (for API/Frontend)
    #    الـ plot_results القديمة كانت بتعمل plt.show() مش شغّالة في بيئة web
    @staticmethod
    def get_plot_data(results_dict: dict, title: str = "ODE Solution") -> dict:
        """
        Returns JSON-serializable chart data so the Frontend can draw the graph.
        results_dict format: { "label": {"T_values": array, "Y_values": array}, ... }
        """
        series = []
        for label, data in results_dict.items():
            series.append({
                "label":    label,
                "t_values": list(data["T_values"]),
                "y_values": list(data["Y_values"]),
            })
        return {
            "title":  title,
            "x_label": "Time t",
            "y_label": "y(t)",
            "series": series,
        }

    # ✅ get_plot_image(): returns a base64 PNG – fallback for non-JS clients
    @staticmethod
    def get_plot_image(results_dict: dict, title: str = "ODE Solution") -> str:
        """Returns a base64-encoded PNG string embeddable as <img src='data:image/png;base64,...'>"""
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, data in results_dict.items():
            ax.plot(data["T_values"], data["Y_values"], label=label)
        ax.set_xlabel("Time t")
        ax.set_ylabel("y(t)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ✅ plot_results() kept for local/CLI usage (calls plt.show())
    @staticmethod
    def plot_results(results_dict, title="ODE Solution"):
        plt.figure(figsize=(8, 5))
        for label, data in results_dict.items():
            plt.plot(data["T_values"], data["Y_values"], label=label)
        plt.xlabel("Time t")
        plt.ylabel("y(t)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    # ✅ Compare two methods
    @staticmethod
    def compare(y1, y2, label1="Method1", label2="Method2"):
        y1, y2 = np.array(y1), np.array(y2)
        min_len = min(len(y1), len(y2))
        error = np.abs(y1[:min_len] - y2[:min_len])
        print(f"Max difference between {label1} and {label2}: {np.max(error)}")
        return error