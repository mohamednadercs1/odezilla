import numpy as np
from typing import Dict, Any

class PhysicsMathModels:
    """
    A production-ready library of physical and mathematical ODE models.
    """
    def __init__(self):
        self.models_catalog = {
            "population_growth":        self._population_growth,
            "logistic_growth":          self._logistic_growth,
            "newton_cooling":           self._newton_cooling,
            "simple_pendulum":          self._simple_pendulum,
            "free_fall_air_resistance": self._free_fall_air_resistance,
            "torricelli_law":           self._torricelli_law
        }

    def get_model(self, model_name: str, **parameters) -> Dict[str, Any]:
        if model_name in self.models_catalog:
            return self.models_catalog[model_name](**parameters)
        else:
            raise ValueError(f"[Error] The model '{model_name}' is not found.")

    # ==========================================
    # 1. Population Growth Model
    # ==========================================
    def _population_growth(self, k: float = 0.1) -> Dict[str, Any]:
        if k < 0:
            raise ValueError("k must be positive.")
        def derivative(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([k * y[0]])
        return {
            "derivative_function": derivative,
            "dimension": 1,
            "recommended_initial": [100.0],
            "title": "Population Growth Model",
            "description": "Exponential growth"
        }

    # ==========================================
    # 2. Logistic Growth Model
    # ==========================================
    def _logistic_growth(self, r: float = 0.1, K: float = 1000.0) -> Dict[str, Any]:
        if K <= 0:
            raise ValueError("K must be > 0.")
        def derivative(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([r * y[0] * (1 - y[0] / K)])
        return {
            "derivative_function": derivative,
            "dimension": 1,
            "recommended_initial": [50.0],
            "title": "Logistic Growth Model",
            "description": "Growth with limit"
        }

    # ==========================================
    # 3. Newton Cooling
    # ==========================================
    def _newton_cooling(self, k: float = 0.05, T_env: float = 25.0) -> Dict[str, Any]:
        def derivative(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([-k * (y[0] - T_env)])
        return {
            "derivative_function": derivative,
            "dimension": 1,
            "recommended_initial": [90.0],
            "title": "Cooling",
            "description": "Temperature change"
        }

    # ==========================================
    # 4. Pendulum
    # ==========================================
    def _simple_pendulum(self, g: float = 9.81, L: float = 1.0) -> Dict[str, Any]:
        if L <= 0:
            raise ValueError("L must be > 0.")
        def derivative(t: float, y: np.ndarray) -> np.ndarray:
            theta, omega = y
            return np.array([omega, -(g / L) * np.sin(theta)])
        return {
            "derivative_function": derivative,
            "dimension": 2,
            "recommended_initial": [0.5, 0.0],
            "title": "Pendulum",
            "description": "Oscillation"
        }

    # ==========================================
    # 5. Free Fall
    # ==========================================
    def _free_fall_air_resistance(self, g: float = 9.81, m: float = 70.0, c: float = 0.25):
        def derivative(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([g - (c / m) * y[0]])
        return {
            "derivative_function": derivative,
            "dimension": 1,
            "recommended_initial": [0.0],
            "title": "Free Fall",
            "description": "Falling object"
        }

    # ==========================================
    # 6. Torricelli
    # ==========================================
    def _torricelli_law(self, A: float = 2.0, a: float = 0.05, g: float = 9.81):
        def derivative(t: float, y: np.ndarray) -> np.ndarray:
            h = max(y[0], 0)
            return np.array([-(a / A) * np.sqrt(2 * g * h)])
        return {
            "derivative_function": derivative,
            "dimension": 1,
            "recommended_initial": [10.0],
            "title": "Tank Drain",
            "description": "Water draining"
        }