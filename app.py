"""
Module 2 – Backend Manager (v4)
FastAPI: symbolic (SymPy) + numerical (Euler / Heun / RK4 / RK45) ODE solving.
"""

from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.integrate import solve_ivp

from modules.engine import SymPyNumericalSolver as NumericalSolver
from modules.models import PhysicsMathModels
from modules.explainer import explain_topic, list_topics
from modules.quiz import QuizEngine
from modules.symbolic_solver import (
    solve_symbolically, to_numeric_function, exact_function,
)

_model_library = PhysicsMathModels()
_MODEL_ALIASES = {
    "population": "population_growth", "population growth": "population_growth",
    "logistic": "logistic_growth", "logistic growth": "logistic_growth",
    "cooling": "newton_cooling", "newton cooling": "newton_cooling",
    "pendulum": "simple_pendulum", "simple pendulum": "simple_pendulum",
    "free fall": "free_fall_air_resistance", "freefall": "free_fall_air_resistance",
    "torricelli": "torricelli_law", "tank drain": "torricelli_law",
}
_TOPIC_ALIASES = {  # so /test-yourself accepts friendly names
    "separable": "separation", "separation of variables": "separation",
    "bernoulli equation": "bernoulli", "exact equation": "exact",
}
_SOLVERS = {"euler", "heun", "rk4", "rk45"}
_EXAMPLES = ["-2*y", "y' + 2*y = x", "dy/dx = x*y", "y' + y = x*y^2",
             "(2*x*y)dx + (x^2)dy = 0", "pendulum", "logistic"]


def _fmt(v):
    return f"{float(v):.6g}"


# ── Physics-model path ───────────────────────────────────────────────────────
def _run_model(problem, model_key, y0, t_start, t_end, solver_type, h):
    model = _model_library.get_model(model_key)
    rec = model["recommended_initial"]
    deriv = model["derivative_function"]

    if model["dimension"] == 1:
        y_init = rec[0] if y0 is None else y0
        fn = lambda t, y: float(deriv(t, np.array([y]))[0])
        res = NumericalSolver(fn, (t_start, t_end), y_init, solver_type, h).solve()
        t_vals, y_vals, method_used, warning = (
            res["T_values"], res["Y_values"], solver_type.upper(), res["warning"])
        init, last = [y_init], y_vals[-1]
    else:
        sol = solve_ivp(deriv, (t_start, t_end), rec, method="RK45",
                        t_eval=np.linspace(t_start, t_end, 200))
        t_vals, y_vals, method_used, warning = sol.t, sol.y, "RK45 (SciPy)", None
        init, last = rec, y_vals[0][-1]

    return {
        "problem": problem, "model": model["title"], "description": model["description"],
        "type": model["title"].lower(), "method": method_used, "initial_value": init,
        "num_points": int(len(t_vals)),
        "answer": f"y({_fmt(t_vals[-1])}) ≈ {last:.4f}",
        "chart_data": NumericalSolver.get_plot_data(
            {model["title"]: {"T_values": t_vals, "Y_values": y_vals}}, title=model["title"]),
        "symbolic_solution": None, "symbolic_steps": [], "warning": warning,
    }


# ── Custom-ODE path ──────────────────────────────────────────────────────────
def _run_custom(problem, subject, level, y0, t_start, t_end, solver_type, h):
    y0 = 1.0 if y0 is None else y0
    sym = solve_symbolically(problem, x0=t_start, y0=y0)
    if sym["numerical_rhs"] is None:           # equation could not even be parsed
        raise ValueError(sym["error"] or "Could not understand the equation.")

    ode_type = sym["ode_type"]
    t_vals = y_vals = chart = warning = num_err = err_vs_exact = None
    try:
        fn = to_numeric_function(problem)
        res = NumericalSolver(fn, (t_start, t_end), y0, solver_type, h).solve()
        t_vals, y_vals, warning = res["T_values"], res["Y_values"], res["warning"]

        series = {f"{solver_type.upper()} (numerical)": {"T_values": t_vals, "Y_values": y_vals}}
        g = exact_function(sym)
        if g is not None:
            ev = g(t_vals)
            if ev is not None and np.all(np.isfinite(ev)):
                series["Exact (symbolic)"] = {"T_values": t_vals, "Y_values": ev}
                err_vs_exact = float(np.max(np.abs(ev - y_vals)))
        chart = NumericalSolver.get_plot_data(series, title=f"Solution: {problem}")
    except Exception as e:
        num_err = str(e)

    steps = [f"Step 1: identified type → {ode_type}"] + sym["steps"]
    if t_vals is not None:
        steps.append(f"Numerical: {solver_type.upper()} with {len(t_vals) - 1} steps, "
                     f"y({_fmt(t_vals[0])}) = {y0:g}")
        steps.append(f"Result: y({_fmt(t_vals[-1])}) ≈ {y_vals[-1]:.4f}")
        if err_vs_exact is not None:
            steps.append(f"Max |numerical − exact| = {err_vs_exact:.3e}")

    answer = (f"y({_fmt(t_vals[-1])}) ≈ {y_vals[-1]:.4f}" if y_vals is not None
              else "Symbolic solution only")
    return {
        "problem": problem, "equation": problem,
        "type": ode_type, "ode_type": ode_type, "method": solver_type.upper(),
        "subject": subject, "level": level,
        "symbolic_solution": sym["symbolic_solution"],
        "particular_solution": sym["particular_solution"],
        "implicit_solution": sym["implicit_solution"],
        "symbolic_steps": sym["steps"], "symbolic_error": sym["error"],
        "dy_dx_form": sym["dy_dx_form"],
        "num_points": int(len(t_vals)) if t_vals is not None else 0,
        "answer": answer, "chart_data": chart,
        "numerical_error": num_err, "error_vs_exact": err_vs_exact,
        "warning": warning, "steps": steps,
    }


def _run_solver(problem, subject, level, y0=None, t_start=0.0, t_end=10.0,
                solver_type="rk4", h=0.1):
    problem = (problem or "").strip()
    if not problem:
        raise ValueError("Please enter an equation.")
    solver_type = (solver_type or "rk4").lower()
    if solver_type not in _SOLVERS:
        raise ValueError(f"Unknown solver '{solver_type}'. Choose from {sorted(_SOLVERS)}.")
    if t_end == t_start:
        raise ValueError("t_end must be different from t_start.")

    key = problem.lower()
    model_key = _MODEL_ALIASES.get(key) or (key if key in _model_library.models_catalog else None)
    if model_key:
        return _run_model(problem, model_key, y0, t_start, t_end, solver_type, h)
    return _run_custom(problem, subject, level, y0, t_start, t_end, solver_type, h)


# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Tutor API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class SolveRequest(BaseModel):
    problem: str
    subject: Optional[str] = "math"
    level: Optional[str] = "high school"
    y0: Optional[float] = None          # None → 1.0 (custom) or the model's default
    t_start: Optional[float] = 0.0
    t_end: Optional[float] = 10.0
    solver_type: Optional[str] = "rk4"  # euler | heun | rk4 | rk45
    h: Optional[float] = 0.1            # step size for fixed-step methods


class ExplainRequest(BaseModel):
    topic: str
    level: Optional[str] = "beginner"
    language: Optional[str] = "arabic"


class QuizRequest(BaseModel):
    topic: str
    num_questions: Optional[int] = 5
    difficulty: Optional[str] = "medium"
    solve_result: Optional[dict] = None


@app.get("/")
def root():
    return {"status": "✅ AI Tutor Backend is running!", "version": "4.0.0",
            "available_models": list(_model_library.models_catalog.keys()),
            "available_topics": list_topics(),
            "available_solvers": sorted(_SOLVERS)}


@app.get("/topics")
def get_topics():
    return {"topics": list_topics()}


@app.post("/solve")
def solve(request: SolveRequest):
    try:
        data = _run_solver(request.problem, request.subject, request.level, request.y0,
                           request.t_start, request.t_end, request.solver_type, request.h)
        return {"success": True, "endpoint": "/solve", "data": data}
    except Exception as e:
        return {"success": False, "endpoint": "/solve",
                "error": f"❌ {e}", "examples": _EXAMPLES}


@app.post("/explain")
def explain(request: ExplainRequest):
    try:
        topic = _TOPIC_ALIASES.get(request.topic.strip().lower(), request.topic)
        return {"success": True, "endpoint": "/explain",
                "data": explain_topic(topic, request.level, request.language)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-yourself")
def test_yourself(request: QuizRequest):
    try:
        topic = _TOPIC_ALIASES.get(request.topic.strip().lower(), request.topic)
        engine = QuizEngine(solution_data=request.solve_result)
        questions = engine.generate_from_topic(
            topic=topic, num_questions=request.num_questions,
            difficulty=request.difficulty, solution_data=request.solve_result)
        return {"success": True, "endpoint": "/test-yourself",
                "data": {"topic": topic, "difficulty": request.difficulty,
                         "questions": questions}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
