"""
Module 2 – Backend Manager (The Manager)
FastAPI server that routes requests to the appropriate modules.

✅ FIX 1 – /solve now returns chart_data (JSON lists) + optional base64 image
✅ FIX 2 – equation classifier detects ODE type from string input
✅ FIX 3 – /test-yourself accepts optional solve_result so quiz questions
            are tailored to what the student actually solved
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

# ============================================================
# ✅ FIX 2 – ODE Equation Classifier
#    Detects the type of a raw equation string so /solve can
#    return a meaningful 'type' field (used by FIX 3 for quiz)
# ============================================================

def classify_equation(eq: str) -> str:
    """
    Returns the ODE type as a string based on the equation pattern.
    Examples:
        "-2*y"              → "linear"
        "y**2 + y"          → "bernoulli"
        "-2*y + np.sin(t)"  → "linear"
        "-k*(y - T)"        → "linear"
    """
    eq_lower = eq.lower().replace(" ", "")

    # Bernoulli: has y raised to power ≠ 0,1
    if re.search(r"y\*\*[2-9]", eq_lower) or re.search(r"y\^[2-9]", eq_lower):
        return "bernoulli"

    # Logistic / nonlinear: y * (something - y) pattern
    if re.search(r"y\s*\*\s*\(", eq_lower) and "1-y" in eq_lower.replace(" ", ""):
        return "logistic growth"

    # Exponential / simple linear: only y (no powers)
    if re.search(r"[+\-\*\/]?\s*y\b", eq_lower) and "y**" not in eq_lower:
        return "linear"

    return "custom"


# ============================================================
# Solver helper
# ============================================================

_model_library = PhysicsMathModels()

_MODEL_ALIASES = {
    "population":          "population_growth",
    "population growth":   "population_growth",
    "logistic":            "logistic_growth",
    "logistic growth":     "logistic_growth",
    "cooling":             "newton_cooling",
    "newton cooling":      "newton_cooling",
    "pendulum":            "simple_pendulum",
    "simple pendulum":     "simple_pendulum",
    "free fall":           "free_fall_air_resistance",
    "freefall":            "free_fall_air_resistance",
    "torricelli":          "torricelli_law",
    "tank drain":          "torricelli_law",
}

def _run_solver(problem: str, subject: str, level: str) -> dict:
    key = problem.strip().lower()
    model_key = _MODEL_ALIASES.get(key) or (key if key in _model_library.models_catalog else None)

    # ── Physics / math model ────────────────────────────────────────────────
    if model_key:
        model   = _model_library.get_model(model_key)
        dim     = model["dimension"]
        y0_list = model["recommended_initial"]

        if dim == 1:
            raw_fn    = model["derivative_function"]
            scalar_fn = lambda t, y: float(raw_fn(t, np.array([y]))[0])
            solver    = NumericalSolver(equation=scalar_fn, t_span=(0, 10),
                                        y0=y0_list[0], method="rk4", h=0.1)
            result    = solver.solve()
            t_vals, y_vals = result["T_values"], result["Y_values"]
            method_used = "RK4"
        else:
            sol = solve_ivp(fun=model["derivative_function"], t_span=(0, 10),
                            y0=y0_list, method="RK45",
                            t_eval=np.linspace(0, 10, 200))
            t_vals, y_vals = sol.t, sol.y[0]
            method_used = "RK45 (SciPy)"

        # ✅ FIX 1 – build chart_data from get_plot_data()
        chart_data = NumericalSolver.get_plot_data(
            {model["title"]: {"T_values": t_vals, "Y_values": y_vals}},
            title=model["title"]
        )

        return {
            "problem":      problem,
            "model":        model["title"],
            "description":  model["description"],
            "type":         model["title"].lower(),   # ← FIX 3: type for quiz
            "method":       method_used,
            "initial_value": y0_list,
            "num_points":   int(len(t_vals)),
            "answer":       f"y(10) ≈ {y_vals[-1]:.4f}",
            "chart_data":   chart_data,               # ✅ FIX 1
        }

    # ── Raw equation string ─────────────────────────────────────────────────
    solver = NumericalSolver(equation=None, t_span=(0, 10), y0=1.0,
                             method="rk4", h=0.1, equation_string=problem)
    result  = solver.solve()
    t_vals  = result["T_values"]
    y_vals  = result["Y_values"]

    # ✅ FIX 2 – classify the equation
    ode_type = classify_equation(problem)

    # ✅ FIX 1 – build chart_data
    chart_data = NumericalSolver.get_plot_data(
        {"RK4 Solution": {"T_values": t_vals, "Y_values": y_vals}},
        title=f"Solution: {problem}"
    )

    return {
        "problem":  problem,
        "equation": problem,
        "type":     ode_type,              # ✅ FIX 2
        "method":   "RK4",
        "subject":  subject,
        "level":    level,
        "steps": [
            "الخطوة 1: تحليل المعادلة",
            f"الخطوة 2: التصنيف → {ode_type}",
            "الخطوة 3: تطبيق طريقة RK4",
            f"الخطوة 4: الحل على الفترة t ∈ [0, 10] بخطوة h=0.1",
            f"الخطوة 5: النتيجة النهائية y ≈ {y_vals[-1]:.4f}",
        ],
        "num_points": int(len(t_vals)),
        "answer":     f"y(10) ≈ {y_vals[-1]:.4f}",
        "chart_data": chart_data,          # ✅ FIX 1
    }


# ============================================================
# 1. إنشاء الـ App
# ============================================================
app = FastAPI(
    title="AI Tutor API",
    description="Backend manager that routes requests to solver, explainer, and quiz modules.",
    version="2.0.0"
)

# ============================================================
# 2. CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3. Pydantic Models
# ============================================================

class SolveRequest(BaseModel):
    problem:  str
    subject:  Optional[str] = "math"
    level:    Optional[str] = "high school"

class ExplainRequest(BaseModel):
    topic:    str
    level:    Optional[str] = "beginner"
    language: Optional[str] = "arabic"

class QuizRequest(BaseModel):
    topic:          str
    num_questions:  Optional[int]  = 5
    difficulty:     Optional[str]  = "medium"
    # ✅ FIX 3 – optional: pass the solver result so quiz is context-aware
    solve_result:   Optional[dict] = None

# ============================================================
# 4. Connector Functions
# ============================================================

def call_solver(data: SolveRequest) -> dict:
    """✅ متوصّل بـ engine.py + models.py | بيرجع chart_data + type للـ quiz"""
    return _run_solver(data.problem, data.subject, data.level)


def call_explainer(data: ExplainRequest) -> dict:
    """✅ متوصّل بـ explainer.py"""
    return explain_topic(data.topic, data.level, data.language)


def call_quiz_generator(data: QuizRequest) -> dict:
    """
    ✅ متوصّل بـ quiz.py
    FIX 3: لو الـ Frontend بعت solve_result، الأسئلة هتكون
    مبنية على المعادلة اللي الطالب حلّها فعلاً
    """
    engine    = QuizEngine(solution_data=data.solve_result)
    questions = engine.generate_from_topic(
        topic=data.topic,
        num_questions=data.num_questions,
        difficulty=data.difficulty,
        solution_data=data.solve_result,
    )
    return {
        "topic":      data.topic,
        "difficulty": data.difficulty,
        "context":    data.solve_result,   # echoed back so Frontend knows
        "questions":  questions,
    }

# ============================================================
# 5. الـ Endpoints
# ============================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "status":           "✅ AI Tutor Backend is running!",
        "version":          "2.0.0",
        "available_models": list(_model_library.models_catalog.keys()),
        "available_topics": list_topics(),
    }


@app.post("/solve")
def solve(request: SolveRequest):
    """
    استقبال مسألة → حلّها عدديًا → رجوع الحل + chart_data + نوع المعادلة

    ✅ FIX 1: الـ response فيه chart_data جاهزة للرسم في الـ Frontend
    ✅ FIX 2: الـ response فيه 'type' بيصنّف نوع المعادلة
    """
    try:
        result = call_solver(request)
        return {"success": True, "endpoint": "/solve", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver error: {str(e)}")


@app.post("/explain")
def explain(request: ExplainRequest):
    """استقبال موضوع → رجوع الشرح + الصيغة + الفيديو"""
    try:
        result = call_explainer(request)
        return {"success": True, "endpoint": "/explain", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainer error: {str(e)}")


@app.post("/test-yourself")
def test_yourself(request: QuizRequest):
    """
    استقبال موضوع → رجوع أسئلة اختيار من متعدد

    ✅ FIX 3: لو بعتّ solve_result في الـ request،
    الأسئلة هتكون مخصّصة للمعادلة اللي الطالب حلّها

    مثال للـ Frontend:
    {
      "topic": "linear",
      "num_questions": 3,
      "solve_result": { "type": "linear", "method": "RK4", "model": "Custom Equation" }
    }
    """
    try:
        result = call_quiz_generator(request)
        return {"success": True, "endpoint": "/test-yourself", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generator error: {str(e)}")


# ============================================================
# 6. تشغيل السيرفر
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )