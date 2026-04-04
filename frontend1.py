import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use("Agg")
import requests
import time
from typing import Optional

# =========================
# Config
# =========================
API_BASE = "http://localhost:8000"   # ← غيّر لو الـ backend على port تاني

st.set_page_config(page_title="ODE AI Solver Pro", layout="wide")

# =========================
# Modern Glassmorphism Style
# =========================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b);
color: #f8f8f8; font-family: 'Roboto', sans-serif; }
h1, h2, h3 { color: #38bdf8; }
.stSidebar { background: rgba(20, 25, 40, 0.85); backdrop-filter:
blur(15px); color: #f8f8f8; font-weight: bold; }
.stButton>button { background: rgba(56,189,248,0.85); color:
#fff; font-weight: bold; border-radius: 15px; padding: 12px 25px;
border:none; box-shadow: 0 0 15px rgba(56,189,248,0.6);
transition:0.3s; }
.stButton>button:hover { background: rgba(14,165,233,0.95);
box-shadow: 0 0 25px rgba(14,165,233,0.8); }
.stTextInput>div>div>input { background: rgba(255,255,255,0.05);
color:#f8f8f8; border-radius:12px; border:none; padding:10px; }
.card { background: rgba(30,41,59,0.75); backdrop-filter:
blur(15px); border-radius:20px; padding:20px; margin:10px 0;
transition: transform 0.3s, box-shadow 0.3s; }
.card:hover { transform: translateY(-5px); box-shadow:0 0 30px
rgba(56,189,248,0.5); }
h4 { color:#38bdf8; }
.chat { background: rgba(255,255,255,0.05); padding:15px;
border-radius:10px; margin-bottom:10px; }
.badge { display:inline-block; background:rgba(56,189,248,0.2);
border:1px solid #38bdf8; border-radius:8px; padding:3px 10px;
font-size:0.8em; color:#38bdf8; margin:2px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helper: call backend
# =========================
def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    """POST to the FastAPI backend. Returns JSON or None on error."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend غير متاح – تأكّد إن السيرفر شغّال على port 8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ خطأ غير متوقع: {e}")
        return None

def api_get(endpoint: str) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# =========================
# Helper: draw chart from backend chart_data
# =========================
def draw_chart(chart_data: dict, title: str = "ODE Solution"):
    """Renders chart from the JSON chart_data returned by /solve."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    colors = ["#38bdf8", "#f472b6", "#34d399", "#fbbf24"]
    for idx, series in enumerate(chart_data.get("series", [])):
        ax.plot(series["t_values"], series["y_values"],
                label=series["label"],
                color=colors[idx % len(colors)], linewidth=2.5)
    ax.set_xlabel(chart_data.get("x_label", "t"), color="#f8f8f8")
    ax.set_ylabel(chart_data.get("y_label", "y(t)"), color="#f8f8f8")
    ax.set_title(title, color="#38bdf8", fontsize=15)
    ax.tick_params(colors="#f8f8f8")
    ax.grid(color="#374151", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#1e293b", labelcolor="#f8f8f8")
    st.pyplot(fig)
    plt.close(fig)

# =========================
# Sidebar
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Solver", "AI Explain", "Graph", "Quiz", "AI Resources"]
)

# Backend health indicator in sidebar
health = api_get("/")
if health:
    st.sidebar.success("✅ Backend متصّل")
    st.sidebar.caption(f"v{health.get('version','?')}")
else:
    st.sidebar.error("❌ Backend غير متاح")

# =========================
# Home
# =========================
if page == "Home":
    st.title("ODE AI Solver Pro")
    st.subheader("Smart Math. Fast Solutions.")

    # Show available models & topics from backend
    if health:
        models  = health.get("available_models", [])
        topics  = health.get("available_topics", [])
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**📐 Built-in Physics Models:**")
            for m in models:
                st.markdown(f'<span class="badge">{m}</span>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown("**📚 Explanation Topics:**")
            for t in topics:
                st.markdown(f'<span class="badge">{t}</span>', unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("""
    <div class="card"><h3>Instant Solver</h3><p>Solve ODEs instantly with a click.</p></div>
    <div class="card"><h3>Interactive Graphs</h3><p>Visualize solutions dynamically with neon-themed graphs.</p></div>
    <div class="card"><h3>AI Explanations</h3><p>Get clear, AI-powered step-by-step explanations.</p></div>
    <div class="card"><h3>Practice Quizzes</h3><p>Test your knowledge with interactive questions.</p></div>
    """, unsafe_allow_html=True)

# =========================
# Solver  ← متوصّل بـ /solve
# =========================
elif page == "Solver":
    st.title("Smart Solver")

    eq_input    = st.text_input(
        "Enter Equation or Model Name",
        placeholder="e.g.  -2*y + np.sin(t)   or   pendulum   or   logistic"
    )
    col1, col2 = st.columns(2)
    with col1:
        y0      = st.slider("Initial value y(0)", -10.0, 10.0, 1.0)
        t_start = st.number_input("Start of t", value=0.0)
    with col2:
        t_end   = st.number_input("End of t", value=10.0)
        solver_type = st.selectbox("Choose Solver", ["RK4", "Euler", "RK45"])

    if st.button("Solve") and eq_input.strip():
        with st.spinner("Solving…"):
            payload = {
                "problem": eq_input.strip(),
                "subject": "math",
                "level":   "high school"
            }
            resp = api_post("/solve", payload)

        if resp and resp.get("success"):
            data = resp["data"]

            # ── Results ─────────────────────────────────────────────────
            st.success(f"✅ Answer: **{data.get('answer', 'N/A')}**")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Method",  data.get("method", "—"))
            col_b.metric("ODE Type", data.get("type", "—"))
            col_c.metric("Points",  data.get("num_points", "—"))

            # ── Steps (if custom equation) ───────────────────────────────
            if "steps" in data:
                with st.expander("📋 Solution Steps"):
                    for step in data["steps"]:
                        st.markdown(f"• {step}")

            # ── Chart from backend chart_data ────────────────────────────
            if "chart_data" in data:
                st.markdown("#### 📊 Solution Graph")
                draw_chart(data["chart_data"],
                           title=data.get("model", data.get("equation", "")))

            # ── Save solve_result in session for Quiz page ───────────────
            st.session_state["solve_result"] = {
                "type":   data.get("type", ""),
                "method": data.get("method", "RK4"),
                "model":  data.get("model", "Custom Equation"),
            }
            st.info("💡 انتقل لصفحة Quiz عشان تتحدّى نفسك على المعادلة دي!")

        # ── Fallback: local numerical solve (offline mode) ──────────────
        elif eq_input.strip():
            st.warning("⚠️ Backend غير متاح – بيحل محليًا بدون تصنيف")
            try:
                eq_str = eq_input.strip()
                t_vals = np.linspace(t_start, t_end, 300)
                h      = (t_end - t_start) / 300
                f_sym  = sp.sympify(eq_str.replace("y(t)", "y").replace("y(x)", "y"))
                t_sym, y_sym = sp.symbols("t y")
                f_num  = sp.lambdify((t_sym, y_sym), f_sym, {"numpy": np, "np": np})

                y_vals = [y0]
                for i in range(1, len(t_vals)):
                    tc, yc = t_vals[i-1], y_vals[-1]
                    if solver_type == "Euler":
                        y_vals.append(yc + h * f_num(tc, yc))
                    else:
                        k1 = f_num(tc,        yc)
                        k2 = f_num(tc + h/2,  yc + h*k1/2)
                        k3 = f_num(tc + h/2,  yc + h*k2/2)
                        k4 = f_num(tc + h,    yc + h*k3)
                        y_vals.append(yc + (h/6)*(k1 + 2*k2 + 2*k3 + k4))

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827")
                ax.plot(t_vals, y_vals, color="#38bdf8", linewidth=2.5)
                ax.set_xlabel("t", color="#f8f8f8"); ax.set_ylabel("y(t)", color="#f8f8f8")
                ax.set_title(f"{solver_type} (local)", color="#38bdf8")
                ax.tick_params(colors="#f8f8f8")
                ax.grid(color="#374151", linestyle="--", linewidth=0.5)
                st.pyplot(fig); plt.close(fig)
            except Exception as ex:
                st.error(f"Local solver error: {ex}")

# باقي الصفحات (AI Explain, Graph, Quiz, AI Resources) نفس الكود، مع تعديل كل `dict | None` → `Optional[dict]`
# فقط استبدل في الدوال `api_post` و `api_get` كما فعلنا أعلاه، وباقي الكود يبقى كما هو.
