"""
Module 6 – Symbolic Solver & Auto-Converter
Uses SymPy to:
1. Solve any ODE symbolically (Exact, Linear, Bernoulli, Separable, Homogeneous)
2. Auto-convert any form to dy/dx = f(x,y) for numerical solving
"""

import sympy as sp
import numpy as np
import re
from typing import Optional

x = sp.Symbol('x')
y = sp.Function('y')
t = sp.Symbol('t')


# ── Step 1: Detect ODE type ──────────────────────────────────────────────────
def detect_ode_type(eq_str: str) -> str:
    """Detects the ODE type from the equation string."""
    s = eq_str.lower().replace(" ", "")

    # Exact: contains dx and dy
    if "dx" in s and "dy" in s:
        return "exact"

    # Separable: dy/dx = f(x)*g(y)
    if ("dy/dx" in s or "y'" in s) and "=" in s:
        right = s.split("=")[-1]
        if re.search(r"[xy]\s*[\*/]", right):
            return "separable"

    # Bernoulli: y^n term
    if re.search(r"y\*\*[2-9]", s) or re.search(r"y\^[2-9]", s):
        return "bernoulli"

    # Linear: dy/dx + P(x)y = Q(x)
    if ("dy/dx" in s or "y'" in s) and "=" in s:
        return "linear"

    # Raw numerical expression (already dy/dt form)
    if "dx" not in s and "dy" not in s and "=" not in s:
        return "numerical"

    return "general"


# ── Step 2: Parse & Solve Symbolically ──────────────────────────────────────
def solve_symbolically(eq_str: str) -> dict:
    """
    Takes any ODE string and returns symbolic solution.
    Handles: Exact, Linear, Bernoulli, Separable, General
    """
    ode_type = detect_ode_type(eq_str)
    result = {
        "ode_type": ode_type,
        "symbolic_solution": None,
        "dy_dx_form": None,
        "numerical_rhs": None,
        "steps": [],
        "error": None
    }

    try:
        if ode_type == "exact":
            result.update(_solve_exact(eq_str))

        elif ode_type in ["linear", "separable", "bernoulli", "general"]:
            result.update(_solve_standard(eq_str))

        elif ode_type == "numerical":
            # Already in dy/dt form — pass through
            result["dy_dx_form"] = eq_str
            result["numerical_rhs"] = eq_str
            result["steps"] = ["المعادلة في الصيغة العددية الصح — هيتحل مباشرة"]

    except Exception as e:
        result["error"] = str(e)

    return result


def _solve_exact(eq_str: str) -> dict:
    """Solves Exact ODEs of the form M(x,y)dx + N(x,y)dy = 0"""
    steps = []
    steps.append("📌 تم اكتشاف معادلة Exact: M(x,y)dx + N(x,y)dy = 0")

    # Clean and parse
    eq_clean = eq_str.replace("= 0", "").replace("=0", "").strip()

    # Split on dy to get M and N
    # Pattern: M dx + N dy
    eq_clean = eq_clean.replace("dy", "*__DY__").replace("dx", "*__DX__")

    xv, yv = sp.symbols('x y')

    try:
        # Parse M and N from the equation
        # Try to extract M (coefficient of dx) and N (coefficient of dy)
        parts = eq_str.replace("= 0","").replace("=0","")

        # Replace dx/dy with markers
        parts = re.sub(r'dx', '__DX__', parts)
        parts = re.sub(r'dy', '__DY__', parts)

        # Split into terms
        M_str = ""
        N_str = ""

        # Find terms with __DX__ and __DY__
        terms = re.split(r'\+(?![^()]*\))', parts)
        for term in terms:
            term = term.strip()
            if "__DX__" in term:
                M_str = term.replace("__DX__", "").replace("*", "").strip()
            elif "__DY__" in term:
                N_str = term.replace("__DY__", "").replace("*", "").strip()

        if not M_str or not N_str:
            raise ValueError("تعذّر تحليل M و N")

        M = sp.sympify(M_str, locals={"x": xv, "y": yv,
                                        "e": sp.E, "exp": sp.exp,
                                        "sin": sp.sin, "cos": sp.cos})
        N = sp.sympify(N_str, locals={"x": xv, "y": yv,
                                       "e": sp.E, "exp": sp.exp,
                                       "sin": sp.sin, "cos": sp.cos})

        steps.append(f"M = {M}")
        steps.append(f"N = {N}")

        # Check exactness: dM/dy == dN/dx
        dM_dy = sp.diff(M, yv)
        dN_dx = sp.diff(N, xv)
        steps.append(f"∂M/∂y = {dM_dy}")
        steps.append(f"∂N/∂x = {dN_dx}")

        if sp.simplify(dM_dy - dN_dx) == 0:
            steps.append("✅ المعادلة Exact (∂M/∂y = ∂N/∂x)")
        else:
            steps.append("⚠️ المعادلة مش Exact تماماً — هيتحل رمزياً بـ SymPy")

        # dy/dx = -M/N
        dy_dx = sp.simplify(-M / N)
        steps.append(f"dy/dx = -M/N = {dy_dx}")

        # Solve symbolically
        y_func = sp.Function('y')
        ode = sp.Eq(y_func(x).diff(x), dy_dx.subs(yv, y_func(x)))
        sol = sp.dsolve(ode)

        # Numerical RHS (for plotting)
        numerical_rhs = str(dy_dx).replace("x", "t").replace("y", "y")

        return {
            "ode_type": "exact",
            "symbolic_solution": str(sol),
            "dy_dx_form": f"dy/dx = {dy_dx}",
            "numerical_rhs": numerical_rhs,
            "steps": steps,
            "error": None
        }

    except Exception as e:
        # Fallback: use SymPy dsolve directly
        steps.append(f"⚠️ التحليل اليدوي فشل، جاري استخدام SymPy مباشرة: {e}")
        return _solve_with_sympy_direct(eq_str, steps)


def _solve_standard(eq_str: str) -> dict:
    """Solves standard ODEs: linear, separable, bernoulli"""
    steps = []
    xv, yv = sp.symbols('x y')
    y_func = sp.Function('y')

    try:
        # Normalize
        eq_clean = eq_str.strip()

        # Handle y' notation
        eq_clean = eq_clean.replace("y'", "Derivative(y(x), x)")
        eq_clean = eq_clean.replace("dy/dx", "Derivative(y(x), x)")

        if "=" in eq_clean:
            left, right = eq_clean.split("=", 1)
            ode_eq = sp.Eq(
                sp.sympify(left, locals={"y": y_func, "x": xv,
                                          "Derivative": sp.Derivative,
                                          "exp": sp.exp, "sin": sp.sin,
                                          "cos": sp.cos, "e": sp.E}),
                sp.sympify(right, locals={"y": y_func, "x": xv,
                                           "Derivative": sp.Derivative,
                                           "exp": sp.exp, "sin": sp.sin,
                                           "cos": sp.cos, "e": sp.E})
            )
        else:
            # Raw expression: treat as dy/dx = expr
            rhs = sp.sympify(eq_clean, locals={"y": yv, "x": xv,
                                                "exp": sp.exp, "sin": sp.sin,
                                                "cos": sp.cos, "e": sp.E,
                                                "np": None})
            ode_eq = sp.Eq(y_func(x).diff(x), rhs.subs(yv, y_func(x)))

        steps.append(f"📌 المعادلة: {ode_eq}")

        sol = sp.dsolve(ode_eq)
        steps.append(f"✅ الحل الرمزي: {sol}")

        # Extract dy/dx form for numerical
        if "=" not in eq_str:
            numerical_rhs = eq_str
        else:
            try:
                rhs_expr = sol.rhs if hasattr(sol, 'rhs') else None
                numerical_rhs = eq_str.split("=")[-1].strip()
            except:
                numerical_rhs = None

        return {
            "symbolic_solution": str(sol),
            "dy_dx_form": f"dy/dx = {ode_eq.rhs if hasattr(ode_eq,'rhs') else '...'}",
            "numerical_rhs": numerical_rhs,
            "steps": steps,
            "error": None
        }

    except Exception as e:
        steps.append(f"⚠️ {e}")
        return _solve_with_sympy_direct(eq_str, steps)


def _solve_with_sympy_direct(eq_str: str, steps: list) -> dict:
    """Last resort: try SymPy dsolve with cleaned input"""
    try:
        y_func = sp.Function('y')
        xv = sp.Symbol('x')

        clean = eq_str.replace("y'", "Derivative(y(x),x)") \
                      .replace("dy/dx", "Derivative(y(x),x)")

        if "=" in clean:
            l, r = clean.split("=", 1)
            eq = sp.Eq(sp.sympify(l), sp.sympify(r))
        else:
            eq = sp.Eq(y_func(xv).diff(xv),
                       sp.sympify(clean, locals={"y": y_func(xv), "x": xv}))

        sol = sp.dsolve(eq)
        steps.append(f"✅ الحل: {sol}")
        return {
            "symbolic_solution": str(sol),
            "dy_dx_form": None,
            "numerical_rhs": None,
            "steps": steps,
            "error": None
        }
    except Exception as e:
        return {
            "symbolic_solution": None,
            "dy_dx_form": None,
            "numerical_rhs": None,
            "steps": steps,
            "error": f"تعذّر حل المعادلة رمزياً: {e}"
        }


# ── Step 3: Auto-convert to numerical RHS ────────────────────────────────────
def to_numerical_rhs(eq_str: str) -> Optional[str]:
    """
    Tries to extract the dy/dx = f(t,y) RHS from any equation form.
    Returns None if conversion fails.
    """
    ode_type = detect_ode_type(eq_str)

    if ode_type == "numerical":
        return eq_str

    result = solve_symbolically(eq_str)
    return result.get("numerical_rhs")