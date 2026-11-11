"""
Module 6 – Symbolic solver (single unified version)

Accepts a first-order ODE in almost any form:
    -2*y                       raw right-hand side  (dy/dx = -2y)
    dy/dx = x*y     y' = x*y   explicit
    y' + 2y = x                 implicit in y'  (solved for y')
    (2xy)dx + (x^2)dy = 0       differential form M dx + N dy = 0

and returns: type, step-by-step explanation, general solution, particular
solution (when an initial condition is given), and a numerical RHS.
"""

import re
import concurrent.futures as cf

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

X = sp.Symbol("x")          # independent variable (t is treated as x)
yv = sp.Symbol("y")         # dependent variable inside f(x, y)
YP = sp.Symbol("YP")        # y'
DX, DY = sp.symbols("DX DY")
Y = sp.Function("y")

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_NS = {
    "x": X, "t": X, "y": yv, "YP": YP, "DX": DX, "DY": DY,
    "e": sp.E, "E": sp.E, "pi": sp.pi,
    "exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log, "abs": sp.Abs,
    "atan": sp.atan, "arctan": sp.atan,
}
SYMPY_TIMEOUT = 15  # seconds


def _s(e):
    return str(e).replace("**", "^")


# ── Parsing ──────────────────────────────────────────────────────────────────
def _preprocess(s: str) -> str:
    s = s.strip().replace("′", "'").replace("’", "'").replace("−", "-")
    if re.search(r"y\s*'\s*'|d\s*\^?\s*2\s*y|d2y", s):
        raise ValueError("Only first-order ODEs are supported.")
    s = re.sub(r"y\s*\(\s*[xt]\s*\)", "y", s)                  # y(x) -> y
    s = re.sub(r"d\s*y\s*/\s*d\s*[xt]", " YP ", s)              # dy/dx -> YP
    s = re.sub(r"y\s*'", " YP ", s)                             # y'    -> YP
    s = re.sub(r"d([xy])(?![A-Za-z0-9_])",
               lambda m: " D" + m.group(1).upper() + " ", s)    # dx, dy -> DX, DY
    return s


def _p(s: str) -> sp.Expr:
    return parse_expr(s, local_dict=dict(_NS), transformations=_TRANSFORMS)


def parse_ode(eq_str: str) -> dict:
    """Returns {"f": f(x,y), "M": M|None, "N": N|None, "differential": bool}."""
    s = _preprocess(eq_str)
    if not s:
        raise ValueError("Empty equation.")
    has_eq = "=" in s
    if has_eq:
        left, right = s.split("=", 1)
        expr = _p(left) - _p(right)
    else:
        expr = _p(s)

    M = N = None
    if expr.has(DX) or expr.has(DY):
        e = sp.expand(expr)
        M, N = e.coeff(DX), e.coeff(DY)
        if sp.simplify(e - M * DX - N * DY) != 0 or N == 0 or M.has(DX, DY) or N.has(DX, DY):
            raise ValueError("Differential form must look like M(x,y) dx + N(x,y) dy = 0.")
        f, differential = sp.simplify(-M / N), True
    else:
        differential = False
        if expr.has(YP):
            sols = sp.solve(expr, YP)
            if len(sols) != 1:
                raise ValueError("Could not solve the equation for y'.")
            f = sols[0]
        elif not has_eq:
            f = expr                       # raw RHS: dy/dx = expr
        else:
            raise ValueError("No derivative found. Write e.g. dy/dx = -2*y or y' + 2y = x.")

    extra = f.free_symbols - {X, yv}
    if extra:
        raise ValueError("Unknown symbol(s): " + ", ".join(sorted(map(str, extra))))
    return {"f": f, "M": M, "N": N, "differential": differential}


def _ode(f):
    return sp.Eq(Y(X).diff(X), f.subs(yv, Y(X)))


# ── Classification (names match explainer.py / quiz.py) ─────────────────────
def classify(parsed: dict) -> str:
    f = parsed["f"]
    if parsed["differential"] and sp.simplify(
            sp.diff(parsed["M"], yv) - sp.diff(parsed["N"], X)) == 0:
        return "exact"
    try:
        hints = sp.classify_ode(_ode(f), Y(X))
    except Exception:
        hints = ()
    if any(h.startswith("separable") for h in hints):
        return "separation"
    if "1st_linear" in hints:
        return "linear"
    if "Bernoulli" in hints:
        return "bernoulli"
    if any("homogeneous_coeff" in h for h in hints):
        return "homogeneous"
    if parsed["differential"]:
        return "non exact"
    if "1st_exact" in hints:
        return "exact"
    return "general"


def detect_ode_type(eq_str: str) -> str:
    try:
        return classify(parse_ode(eq_str))
    except Exception:
        return "general"


# ── Step-by-step explanations ───────────────────────────────────────────────
def _exact_steps(M, N):
    My, Nx = sp.diff(M, yv), sp.diff(N, X)
    steps = [f"M(x,y) = {_s(M)},  N(x,y) = {_s(N)}",
             f"∂M/∂y = {_s(My)},  ∂N/∂x = {_s(Nx)}"]
    if sp.simplify(My - Nx) != 0:
        return steps, None
    steps.append("∂M/∂y = ∂N/∂x, so the equation is exact.")
    psi1 = sp.integrate(M, X)
    gp = sp.simplify(N - sp.diff(psi1, yv))
    g = sp.integrate(gp, yv)
    psi = sp.simplify(psi1 + g)
    steps += [f"ψ(x,y) = ∫M dx = {_s(psi1)} + g(y)",
              f"g'(y) = N − ∂/∂y∫M dx = {_s(gp)}  →  g(y) = {_s(g)}",
              f"Implicit solution: {_s(psi)} = C"]
    return steps, psi


def _method_steps(kind, parsed):
    """Returns (steps, implicit_solution_string_or_None). Never raises."""
    f = parsed["f"]
    steps, implicit = [], None
    try:
        if kind == "separation":
            d = sp.separatevars(f, dict=True, symbols=[X, yv])
            g, h = d["coeff"] * d.get(X, 1), d.get(yv, 1)
            steps = [f"Write dy/dx = g(x)·h(y) with g(x) = {_s(g)}, h(y) = {_s(h)}.",
                     "Separate the variables: dy / h(y) = g(x) dx.",
                     f"Integrate: ∫ 1/({_s(h)}) dy = ∫ ({_s(g)}) dx.",
                     f"Left: {_s(sp.integrate(1 / h, yv))};  right: {_s(sp.integrate(g, X))} + C.",
                     "Solve for y and use the initial condition to find C."]

        elif kind == "linear":
            Q = sp.simplify(f.subs(yv, 0))
            P = sp.simplify(-sp.diff(f, yv))
            mu = sp.simplify(sp.exp(sp.integrate(P, X)))
            steps = [f"Standard form y' + P(x)y = Q(x) with P = {_s(P)}, Q = {_s(Q)}.",
                     f"Integrating factor μ(x) = exp(∫P dx) = {_s(mu)}.",
                     f"Multiply through: (μ·y)' = μ·Q = {_s(sp.simplify(mu * Q))}.",
                     f"Integrate: μ·y = {_s(sp.integrate(sp.simplify(mu * Q), X))} + C.",
                     "Divide by μ and use the initial condition to find C."]

        elif kind == "bernoulli":
            fe = sp.expand(f)
            exps = {t.as_coeff_exponent(yv)[1] for t in sp.Add.make_args(fe)}
            n = [e for e in exps if e not in (0, 1)][0]
            P, Q = -fe.coeff(yv, 1), fe.coeff(yv, n)
            steps = [f"Bernoulli form y' + P(x)y = Q(x)yⁿ with P = {_s(P)}, Q = {_s(Q)}, n = {n}.",
                     f"Substitute v = y^(1−n) = y^{1 - n}; then v' = (1−n)·y^(−n)·y'.",
                     f"This gives the linear ODE v' + (1−n)P v = (1−n)Q, i.e. "
                     f"v' + {_s(sp.simplify((1 - n) * P))}·v = {_s(sp.simplify((1 - n) * Q))}.",
                     "Solve for v with an integrating factor, then back-substitute y = v^(1/(1−n))."]

        elif kind == "homogeneous":
            v = sp.Symbol("v")
            F = sp.simplify(f.subs(yv, v * X))
            if F.has(X):
                raise ValueError("not homogeneous")
            steps = [f"dy/dx = F(y/x) with F(v) = {_s(F)}.",
                     "Substitute y = v·x, so y' = v + x·v'.",
                     f"Then x·v' = F(v) − v = {_s(sp.simplify(F - v))}.",
                     "Separate: dv / (F(v) − v) = dx / x, and integrate.",
                     "Back-substitute v = y/x."]

        elif kind == "exact":
            M, N = (parsed["M"], parsed["N"]) if parsed["differential"] else (-f, sp.Integer(1))
            steps, psi = _exact_steps(M, N)
            implicit = f"{_s(psi)} = C" if psi is not None else None

        elif kind == "non exact":
            M, N = parsed["M"], parsed["N"]
            My, Nx = sp.diff(M, yv), sp.diff(N, X)
            steps = [f"M = {_s(M)}, N = {_s(N)}; ∂M/∂y = {_s(My)} ≠ ∂N/∂x = {_s(Nx)}: not exact."]
            a = sp.simplify((My - Nx) / N)
            if not a.has(yv):
                mu = sp.simplify(sp.exp(sp.integrate(a, X)))
                steps.append(f"Integrating factor depends on x only: μ(x) = exp(∫(M_y−N_x)/N dx) = {_s(mu)}.")
            else:
                b = sp.simplify((Nx - My) / M)
                if b.has(X):
                    raise ValueError("no simple integrating factor")
                mu = sp.simplify(sp.exp(sp.integrate(b, yv)))
                steps.append(f"Integrating factor depends on y only: μ(y) = exp(∫(N_x−M_y)/M dy) = {_s(mu)}.")
            M2, N2 = sp.simplify(mu * M), sp.simplify(mu * N)
            steps.append("Multiply M and N by μ; the new equation is exact:")
            more, psi = _exact_steps(M2, N2)
            steps += more
            implicit = f"{_s(psi)} = C" if psi is not None else None
        else:
            steps = ["No special textbook method recognised; SymPy's general ODE solver is used."]
    except Exception:
        steps = steps or ["(Detailed steps unavailable for this form; SymPy solved it directly.)"]
    return steps, implicit


# ── Public: symbolic solution ───────────────────────────────────────────────
def _run(fn, timeout=SYMPY_TIMEOUT):
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except cf.TimeoutError:
        raise TimeoutError(f"SymPy took longer than {timeout}s.")
    finally:
        ex.shutdown(wait=False)


def solve_symbolically(eq_str: str, x0=None, y0=None) -> dict:
    """JSON-safe result (plus a private '_particular' sympy object, popped by app.py)."""
    result = {"ode_type": None, "symbolic_solution": None, "particular_solution": None,
              "implicit_solution": None, "dy_dx_form": None, "numerical_rhs": None,
              "steps": [], "error": None, "_particular": None}
    try:
        parsed = parse_ode(eq_str)
    except Exception as e:
        result["error"] = f"Could not parse equation: {e}"
        return result

    f = parsed["f"]
    kind = classify(parsed)
    steps, implicit = _method_steps(kind, parsed)
    result.update(ode_type=kind, dy_dx_form=f"dy/dx = {_s(f)}",
                  numerical_rhs=str(f), steps=steps, implicit_solution=implicit)
    ode = _ode(f)

    try:
        sol = _run(lambda: sp.dsolve(ode, Y(X)))
        sols = sol if isinstance(sol, list) else [sol]
        result["symbolic_solution"] = "; ".join(_s(s) for s in sols)
        result["steps"].append(f"General solution: {result['symbolic_solution']}")

        if x0 is not None and y0 is not None:
            try:
                ics = {Y(sp.nsimplify(x0, rational=True)): sp.nsimplify(y0, rational=True)}
                psol = _run(lambda: sp.dsolve(ode, Y(X), ics=ics))
                psol = psol[0] if isinstance(psol, list) else psol
                result["particular_solution"] = _s(psol)
                result["_particular"] = psol
                result["steps"].append(f"With y({x0:g}) = {y0:g}:  {result['particular_solution']}")
            except Exception as e:
                result["steps"].append(f"(Could not fit the initial condition symbolically: {e})")
    except Exception as e:
        if not implicit:
            result["error"] = f"No closed-form solution found ({e}). Numerical solution only."
    return result


def to_numeric_function(eq_str: str):
    """Returns f(t, y) -> float for the numerical solvers."""
    f = parse_ode(eq_str)["f"]
    fn = sp.lambdify((X, yv), f, modules=["numpy"])
    return lambda t, y: float(fn(t, y))


def to_numerical_rhs(eq_str: str):
    try:
        return str(parse_ode(eq_str)["f"])
    except Exception:
        return None


def exact_function(result: dict):
    """Callable ts -> exact y(ts) from the particular solution, or None."""
    sol = result.get("_particular")
    try:
        if sol is None or sol.lhs != Y(X) or not sol.rhs.free_symbols <= {X}:
            return None
        fn = sp.lambdify(X, sol.rhs, "numpy")

        def g(ts):
            ts = np.asarray(ts, dtype=float)
            with np.errstate(all="ignore"):
                v = np.broadcast_to(np.asarray(fn(ts), dtype=complex), ts.shape)
            return None if np.max(np.abs(v.imag)) > 1e-9 else v.real
        return g
    except Exception:
        return None
