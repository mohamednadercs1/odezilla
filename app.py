"""
Module 2 – Backend Manager
FastAPI with Symbolic + Numerical solving for any ODE form.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import re
import numpy as np
from scipy.integrate import solve_ivp

from modules.engine import NumericalSolver
from modules.models import PhysicsMathModels
from modules.explainer import explain_topic, list_topics
from modules.quiz import QuizEngine
try:
    from modules.symbolic_solver import solve_symbolically, detect_ode_type, to_numerical_rhs
except ImportError:
    from modules.Symbolic_solver import solve_symbolically, detect_ode_type, to_numerical_rhs

_model_library = PhysicsMathModels()
_MODEL_ALIASES = {
    "population":        "population_growth",
    "population growth": "population_growth",
    "logistic":          "logistic_growth",
    "logistic growth":   "logistic_growth",
    "cooling":           "newton_cooling",
    "newton cooling":    "newton_cooling",
    "pendulum":          "simple_pendulum",
    "simple pendulum":   "simple_pendulum",
    "free fall":         "free_fall_air_resistance",
    "freefall":          "free_fall_air_resistance",
    "torricelli":        "torricelli_law",
    "tank drain":        "torricelli_law",
}

def classify_equation(eq: str) -> str:
    eq_lower = eq.lower().replace(" ", "")
    if re.search(r"y\*\*[2-9]", eq_lower): return "bernoulli"
    if re.search(r"y\s*\*\s*\(", eq_lower) and "1-y" in eq_lower: return "logistic growth"
    if re.search(r"[+\-\*\/]?\s*y\b", eq_lower) and "y**" not in eq_lower: return "linear"
    return "custom"


def _run_solver(problem, subject, level, y0=1.0,
                t_start=0.0, t_end=10.0, solver_type="rk4"):

    key = problem.strip().lower()
    model_key = _MODEL_ALIASES.get(key) or (
        key if key in _model_library.models_catalog else None)

    # ── Physics model ─────────────────────────────────────────────────────
    if model_key:
        model   = _model_library.get_model(model_key)
        dim     = model["dimension"]
        y0_list = [y0] if dim == 1 else model["recommended_initial"]

        if dim == 1:
            raw_fn    = model["derivative_function"]
            scalar_fn = lambda t, y: float(raw_fn(t, np.array([y]))[0])
            solver    = NumericalSolver(equation=scalar_fn,
                                        t_span=(t_start, t_end),
                                        y0=y0_list[0],
                                        method=solver_type.lower(), h=0.1)
            result    = solver.solve()
            t_vals, y_vals = result["T_values"], result["Y_values"]
            method_used = solver_type.upper()
        else:
            sol = solve_ivp(fun=model["derivative_function"],
                            t_span=(t_start, t_end),
                            y0=model["recommended_initial"],
                            method="RK45",
                            t_eval=np.linspace(t_start, t_end, 200))
            t_vals, y_vals = sol.t, sol.y[0]
            method_used = "RK45 (SciPy)"

        chart_data = NumericalSolver.get_plot_data(
            {model["title"]: {"T_values": t_vals, "Y_values": y_vals}},
            title=model["title"]
        )
        return {
            "problem":           problem,
            "model":             model["title"],
            "description":       model["description"],
            "type":              model["title"].lower(),
            "method":            method_used,
            "initial_value":     y0_list,
            "num_points":        int(len(t_vals)),
            "answer":            f"y({t_end}) ≈ {y_vals[-1]:.4f}",
            "chart_data":        chart_data,
            "symbolic_solution": None,
            "symbolic_steps":    [],
        }

    # ── Symbolic + Numerical ──────────────────────────────────────────────
    ode_type     = detect_ode_type(problem)
    symbolic     = solve_symbolically(problem)
    sym_solution = symbolic.get("symbolic_solution")
    sym_steps    = symbolic.get("steps", [])
    sym_error    = symbolic.get("error")

    # Try to get numerical RHS
    numerical_rhs = symbolic.get("numerical_rhs") or problem

    # Try numerical solve
    t_vals = y_vals = None
    chart_data = None
    numerical_error = None

    try:
        solver = NumericalSolver(
            equation=None,
            t_span=(t_start, t_end),
            y0=y0,
            method=solver_type.lower(), h=0.1,
            equation_string=numerical_rhs
        )
        result = solver.solve()
        t_vals = result["T_values"]
        y_vals = result["Y_values"]
        chart_data = NumericalSolver.get_plot_data(
            {f"{solver_type.upper()} Solution": {"T_values": t_vals, "Y_values": y_vals}},
            title=f"Solution: {problem}"
        )
    except Exception as e:
        numerical_error = str(e)

    classified = classify_equation(numerical_rhs) if numerical_rhs else ode_type

    steps = [
        f"الخطوة 1: تحديد نوع المعادلة → {ode_type}",
        f"الخطوة 2: الحل الرمزي بـ SymPy",
    ]
    if sym_solution:
        steps.append(f"الخطوة 3: الحل = {sym_solution}")
    if t_vals is not None:
        steps.append(f"الخطوة 4: تطبيق {solver_type.upper()} عددياً")
        steps.append(f"الخطوة 5: y({t_end}) ≈ {y_vals[-1]:.4f}")

    return {
        "problem":           problem,
        "equation":          problem,
        "type":              classified,
        "ode_type":          ode_type,
        "method":            solver_type.upper(),
        "subject":           subject,
        "level":             level,
        "symbolic_solution": sym_solution,
        "symbolic_steps":    sym_steps,
        "symbolic_error":    sym_error,
        "dy_dx_form":        symbolic.get("dy_dx_form"),
        "num_points":        int(len(t_vals)) if t_vals is not None else 0,
        "answer":            f"y({t_end}) ≈ {y_vals[-1]:.4f}" if y_vals is not None else "الحل الرمزي فقط",
        "chart_data":        chart_data,
        "numerical_error":   numerical_error,
        "steps":             steps,
    }


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Tutor API", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class SolveRequest(BaseModel):
    problem:     str
    subject:     Optional[str]   = "math"
    level:       Optional[str]   = "high school"
    y0:          Optional[float] = 1.0
    t_start:     Optional[float] = 0.0
    t_end:       Optional[float] = 10.0
    solver_type: Optional[str]   = "rk4"

class ExplainRequest(BaseModel):
    topic:    str
    level:    Optional[str] = "beginner"
    language: Optional[str] = "arabic"

class QuizRequest(BaseModel):
    topic:         str
    num_questions: Optional[int]  = 5
    difficulty:    Optional[str]  = "medium"
    solve_result:  Optional[dict] = None

@app.get("/")
def root():
    return {
        "status":           "✅ AI Tutor Backend is running!",
        "version":          "3.0.0",
        "available_models": list(_model_library.models_catalog.keys()),
        "available_topics": list_topics(),
    }

@app.get("/topics")
def get_topics():
    return {"topics": list_topics()}

@app.post("/solve")
def solve(request: SolveRequest):
    try:
        result = _run_solver(
            problem=request.problem, subject=request.subject,
            level=request.level, y0=request.y0,
            t_start=request.t_start, t_end=request.t_end,
            solver_type=request.solver_type,
        )
        return {"success": True, "endpoint": "/solve", "data": result}
    except Exception as e:
        return {
            "success": False,
            "endpoint": "/solve",
            "error": f"❌ خطأ: {str(e)}",
            "examples": ["-2*y", "pendulum", "cooling", "logistic"]
        }

@app.post("/explain")
def explain(request: ExplainRequest):
    try:
        result = explain_topic(request.topic, request.level, request.language)
        return {"success": True, "endpoint": "/explain", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-yourself")
def test_yourself(request: QuizRequest):
    try:
        engine    = QuizEngine(solution_data=request.solve_result)
        questions = engine.generate_from_topic(
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            solution_data=request.solve_result,
        )
        return {"success": True, "endpoint": "/test-yourself",
                "data": {"topic": request.topic, "difficulty": request.difficulty,
                         "questions": questions}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)