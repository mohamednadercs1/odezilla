import sympy as sp
import numpy as np
import re
from typing import Optional

def detect_ode_type(eq_str: str) -> str:
    s = eq_str.lower().replace(" ", "")
    if "dx" in s and "dy" in s:
        return "exact"
    if re.search(r"y\*\*[2-9]", s):
        return "bernoulli"
    if ("dy/dx" in s or "y'" in s) and "=" in s:
        return "linear"
    if "dx" not in s and "dy" not in s and "=" not in s:
        return "numerical"
    return "general"

def solve_symbolically(eq_str: str) -> dict:
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
        xv = sp.Symbol('x')
        yv = sp.Symbol('y')
        y_func = sp.Function('y')

        if ode_type == "numerical":
            result["numerical_rhs"] = eq_str
            result["steps"] = ["المعادلة في الصيغة العددية الصح"]
            return result

        clean = eq_str.strip()
        clean = clean.replace("y'", "Derivative(y(x),x)")
        clean = clean.replace("dy/dx", "Derivative(y(x),x)")

        local_dict = {
            "x": xv, "y": y_func, "Derivative": sp.Derivative,
            "exp": sp.exp, "sin": sp.sin, "cos": sp.cos,
            "e": sp.E, "sqrt": sp.sqrt, "log": sp.log
        }

        if "=" in clean and "dx" not in clean and "dy" not in clean:
            left, right = clean.split("=", 1)
            eq = sp.Eq(sp.sympify(left, locals=local_dict),
                       sp.sympify(right, locals=local_dict))
        elif "dx" in clean and "dy" in clean:
            clean2 = clean.replace("= 0","").replace("=0","")
            clean2 = re.sub(r'dx', '*__DX__', clean2)
            clean2 = re.sub(r'dy', '*__DY__', clean2)
            local2 = {"x": xv, "y": yv, "exp": sp.exp,
                      "sin": sp.sin, "cos": sp.cos, "e": sp.E}
            terms = re.split(r'\+', clean2)
            M_str, N_str = "", ""
            for term in terms:
                term = term.strip()
                if "__DX__" in term:
                    M_str = term.replace("__DX__","").replace("*","").strip()
                elif "__DY__" in term:
                    N_str = term.replace("__DY__","").replace("*","").strip()
            if M_str and N_str:
                M = sp.sympify(M_str, locals=local2)
                N = sp.sympify(N_str, locals=local2)
                dy_dx = sp.simplify(-M / N)
                result["dy_dx_form"] = f"dy/dx = {dy_dx}"
                result["numerical_rhs"] = str(dy_dx).replace("x","t")
                result["steps"].append(f"M = {M}, N = {N}")
                result["steps"].append(f"dy/dx = -M/N = {dy_dx}")
                eq = sp.Eq(y_func(xv).diff(xv), dy_dx.subs(yv, y_func(xv)))
            else:
                raise ValueError("تعذّر تحليل M و N")
        else:
            rhs = sp.sympify(clean, locals={"y": yv, "x": xv,
                             "exp": sp.exp, "sin": sp.sin, "cos": sp.cos})
            eq = sp.Eq(y_func(xv).diff(xv), rhs.subs(yv, y_func(xv)))
            result["numerical_rhs"] = clean

        sol = sp.dsolve(eq)
        result["symbolic_solution"] = str(sol)
        result["steps"].append(f"الحل الرمزي: {sol}")

    except Exception as e:
        result["error"] = str(e)

    return result

def to_numerical_rhs(eq_str: str) -> Optional[str]:
    ode_type = detect_ode_type(eq_str)
    if ode_type == "numerical":
        return eq_str
    result = solve_symbolically(eq_str)
    return result.get("numerical_rhs")
