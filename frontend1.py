import streamlit as st
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import time

API_BASE = "https://web-production-35a16.up.railway.app"

st.set_page_config(page_title="ODE AI Solver Pro", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0f172a,#1e293b);color:#f8f8f8;font-family:'Roboto',sans-serif;}
h1,h2,h3{color:#38bdf8;}
.stSidebar{background:rgba(20,25,40,0.85);backdrop-filter:blur(15px);color:#f8f8f8;font-weight:bold;}
.stButton>button{background:rgba(56,189,248,0.85);color:#fff;font-weight:bold;border-radius:15px;padding:12px 25px;border:none;box-shadow:0 0 15px rgba(56,189,248,0.6);transition:0.3s;}
.stButton>button:hover{background:rgba(14,165,233,0.95);box-shadow:0 0 25px rgba(14,165,233,0.8);}
.stTextInput>div>div>input{background:rgba(255,255,255,0.05);color:#f8f8f8;border-radius:12px;border:none;padding:10px;}
.card{background:rgba(30,41,59,0.75);backdrop-filter:blur(15px);border-radius:20px;padding:20px;margin:10px 0;transition:transform 0.3s,box-shadow 0.3s;}
.card:hover{transform:translateY(-5px);box-shadow:0 0 30px rgba(56,189,248,0.5);}
h4{color:#38bdf8;}
.chat{background:rgba(255,255,255,0.05);padding:15px;border-radius:10px;margin-bottom:10px;}
.badge{display:inline-block;background:rgba(56,189,248,0.2);border:1px solid #38bdf8;border-radius:8px;padding:3px 10px;font-size:0.8em;color:#38bdf8;margin:2px;}
.result-box{background:rgba(56,189,248,0.1);border:1px solid #38bdf8;border-radius:15px;padding:20px;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend offline – make sure uvicorn is running on port 8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def draw_chart(chart_data, title="ODE Solution"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    colors = ["#38bdf8","#f472b6","#34d399","#fbbf24"]
    for idx, series in enumerate(chart_data.get("series", [])):
        ax.plot(series["t_values"], series["y_values"],
                label=series["label"],
                color=colors[idx % len(colors)], linewidth=2.5)
    ax.set_xlabel(chart_data.get("x_label","t"), color="#f8f8f8")
    ax.set_ylabel(chart_data.get("y_label","y(t)"), color="#f8f8f8")
    ax.set_title(title, color="#38bdf8", fontsize=15)
    ax.tick_params(colors="#f8f8f8")
    ax.grid(color="#374151", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#1e293b", labelcolor="#f8f8f8")
    st.pyplot(fig)
    plt.close(fig)

# =========================
# Sidebar
# =========================
page = st.sidebar.radio("Navigation",
    ["Home","Solver","AI Explain","Graph","Quiz","AI Resources"])

health = api_get("/")
if health:
    st.sidebar.success("✅ Backend connected")
    st.sidebar.caption(f"v{health.get('version','?')}")
else:
    st.sidebar.error("❌ Backend offline")

# =========================
# Home
# =========================
if page == "Home":
    st.title("🧮 ODE AI Solver Pro")
    st.subheader("Smart Math. Fast Solutions.")
    if health:
        models = health.get("available_models", [])
        topics = health.get("available_topics", [])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📐 Built-in Models:**")
            for m in models:
                st.markdown(f'<span class="badge">{m}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown("**📚 Topics:**")
            for t in topics:
                st.markdown(f'<span class="badge">{t}</span>', unsafe_allow_html=True)
        st.markdown("---")
    st.markdown("""
    <div class="card"><h3>⚡ Instant Solver</h3><p>Solve ODEs using Euler, RK4, or RK45 with your own initial conditions.</p></div>
    <div class="card"><h3>📊 Interactive Graphs</h3><p>Visualize solutions with neon-themed charts.</p></div>
    <div class="card"><h3>🤖 AI Explanations</h3><p>Get step-by-step explanations for any ODE topic.</p></div>
    <div class="card"><h3>🎯 Practice Quizzes</h3><p>Test yourself with questions tailored to what you just solved.</p></div>
    """, unsafe_allow_html=True)

# =========================
# Solver
# =========================
elif page == "Solver":
    st.title("⚡ Smart Solver")
    st.markdown("""<div class="card"><b>Examples:</b>
    <code>-2*y</code> &nbsp;|&nbsp; <code>-2*y + np.sin(t)</code> &nbsp;|&nbsp;
    <code>pendulum</code> &nbsp;|&nbsp; <code>logistic</code> &nbsp;|&nbsp; <code>cooling</code>
    </div>""", unsafe_allow_html=True)

    eq_input = st.text_input("Enter Equation or Model Name",
                              placeholder="-2*y + np.sin(t)")
    c1, c2 = st.columns(2)
    with c1:
        y0      = st.slider("Initial value y(0)", -10.0, 10.0, 1.0)
        t_start = st.number_input("Start of t", value=0.0)
    with c2:
        t_end       = st.number_input("End of t", value=10.0)
        solver_type = st.selectbox("Solver Method", ["RK4","Euler","RK45"])

    if st.button("🚀 Solve") and eq_input.strip():
        with st.spinner("Solving…"):
            resp = api_post("/solve", {
                "problem":     eq_input.strip(),
                "subject":     "math",
                "level":       "high school",
                "y0":          y0,
                "t_start":     t_start,
                "t_end":       t_end,
                "solver_type": solver_type,
            })
        if resp and not resp.get("success"):
            # ✅ Show validation / solver error clearly
            st.error(resp.get("error", "❌ حصل خطأ"))
            examples = resp.get("examples", [])
            if examples:
                st.markdown("**جرّب الأمثلة دي:**")
                cols = st.columns(len(examples))
                for i, ex in enumerate(examples):
                    cols[i].code(ex)

        elif resp and resp.get("success"):
            data = resp["data"]
            st.markdown(f'<div class="result-box"><h4>✅ {data.get("answer","N/A")}</h4></div>',
                        unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            ca.metric("Method",   data.get("method","—"))
            cb.metric("ODE Type", data.get("type","—"))
            cc.metric("Points",   data.get("num_points","—"))
            if "steps" in data:
                with st.expander("📋 Solution Steps"):
                    for s in data["steps"]:
                        st.markdown(f"• {s}")
            if data.get("chart_data"):               
                st.markdown("#### 📊 Solution Graph")
                draw_chart(data["chart_data"],
                           title=data.get("model", data.get("equation","")))
            st.session_state["solve_result"] = {
                "type":   data.get("type",""),
                "method": data.get("method","RK4"),
                "model":  data.get("model","Custom Equation"),
            }
            st.info("💡 Go to Quiz to test yourself on this equation!")

        elif eq_input.strip():
            st.warning("⚠️ Backend offline – solving locally")
            try:
                t_vals = np.linspace(t_start, t_end, 300)
                h = (t_end - t_start) / 300
                t_sym, y_sym = sp.symbols("t y")
                f_sym = sp.sympify(eq_input.replace("y(t)","y").replace("np.",""))
                f_num = sp.lambdify((t_sym, y_sym), f_sym, {"numpy":np,"np":np})
                y_vals = [y0]
                for i in range(1, len(t_vals)):
                    tc, yc = t_vals[i-1], y_vals[-1]
                    if solver_type == "Euler":
                        y_vals.append(yc + h * f_num(tc, yc))
                    else:
                        k1 = f_num(tc,      yc)
                        k2 = f_num(tc+h/2,  yc+h*k1/2)
                        k3 = f_num(tc+h/2,  yc+h*k2/2)
                        k4 = f_num(tc+h,    yc+h*k3)
                        y_vals.append(yc + (h/6)*(k1+2*k2+2*k3+k4))
                fig, ax = plt.subplots(figsize=(9,4))
                ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827")
                ax.plot(t_vals, y_vals, color="#38bdf8", linewidth=2.5)
                ax.set_xlabel("t",color="#f8f8f8"); ax.set_ylabel("y(t)",color="#f8f8f8")
                ax.set_title(f"{solver_type} – local", color="#38bdf8")
                ax.tick_params(colors="#f8f8f8")
                ax.grid(color="#374151",linestyle="--",linewidth=0.5)
                st.pyplot(fig); plt.close(fig)
            except Exception as ex:
                st.error(f"Local solver error: {ex}")

# =========================
# AI Explain
# =========================
elif page == "AI Explain":
    st.title("🤖 AI Explanation")
    topics_resp = api_get("/topics") if health else None
    topics_list = topics_resp["topics"] if topics_resp else [
        "separation","linear","bernoulli","exact","homogeneous","non homogeneous","non exact"]
    topic_input = st.selectbox("Choose a Topic", topics_list)
    c1, c2 = st.columns(2)
    with c1:
        level = st.selectbox("Level", ["beginner","intermediate","advanced"])
    with c2:
        language = st.selectbox("Language", ["arabic","english"])
    if st.button("💡 Explain"):
        with st.spinner("Getting explanation…"):
            resp = api_post("/explain", {"topic":topic_input,"level":level,"language":language})
        if resp and resp.get("success"):
            data = resp["data"]
            if "error" in data:
                st.error(data["error"])
            else:
                st.markdown(f"""
                <div class="card">
                    <h4>📚 {data['topic'].title()}</h4>
                    <p>{data['explanation']}</p>
                    <hr style="border-color:#334155">
                    <p><b>📐 Formula:</b></p>
                    <p style="font-size:1.2em;color:#34d399;font-family:monospace">{data['formula']}</p>
                    <hr style="border-color:#334155">
                    <p><b>🎥 Video:</b> <a href="{data['video']}" target="_blank"
                    style="color:#38bdf8;">Watch here →</a></p>
                </div>""", unsafe_allow_html=True)

# =========================
# Graph
# =========================
elif page == "Graph":
    st.title("📈 Graph Tool")
    expr = st.text_input("Enter function", placeholder="sin(x), exp(-x), x**2")
    c1, c2 = st.columns(2)
    with c1:
        x_min = st.number_input("x min", value=-5.0)
    with c2:
        x_max = st.number_input("x max", value=5.0)
    if st.button("📊 Plot") and expr.strip():
        try:
            x_sym  = sp.symbols("x")
            f_expr = sp.sympify(expr)
            f_num  = sp.lambdify(x_sym, f_expr, "numpy")
            x_vals = np.linspace(x_min, x_max, 400)
            y_vals = np.array(f_num(x_vals), dtype=float)
            fig, ax = plt.subplots(figsize=(9,4))
            ax.set_facecolor("#111827"); fig.patch.set_facecolor("#111827")
            ax.plot(x_vals, y_vals, color="#38bdf8", linewidth=2.5)
            ax.set_xlim(x_min, x_max)
            ymin=float(np.nanmin(y_vals)); ymax=float(np.nanmax(y_vals))
            ax.set_ylim(ymin-abs(ymin)*0.1-1, ymax+abs(ymax)*0.1+1)
            ax.set_title(f"y = {expr}", color="#38bdf8", fontsize=16)
            ax.set_xlabel("x",color="#f8f8f8"); ax.set_ylabel("y",color="#f8f8f8")
            ax.tick_params(colors="#f8f8f8")
            ax.grid(color="#374151",linestyle="--",linewidth=0.5)
            ax.axhline(0,color="#6b7280",linewidth=0.8)
            ax.axvline(0,color="#6b7280",linewidth=0.8)
            st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# Quiz
# =========================
elif page == "Quiz":
    st.title("🎯 Quick Quiz")
    solve_result = st.session_state.get("solve_result", None)
    if solve_result:
        st.info(f"📌 Questions based on: **{solve_result.get('type','—')}** "
                f"| Method: **{solve_result.get('method','—')}**")
    topics_resp = api_get("/topics") if health else None
    topics_list = topics_resp["topics"] if topics_resp else [
        "linear","bernoulli","exact","separation","homogeneous"]
    default_idx = 0
    if solve_result and solve_result.get("type","") in topics_list:
        default_idx = topics_list.index(solve_result["type"])
    topic_sel = st.selectbox("Topic", topics_list, index=default_idx)
    c1, c2 = st.columns(2)
    with c1:
        n_qs = st.slider("Number of questions", 1, 10, 3)
    with c2:
        difficulty = st.selectbox("Difficulty", ["easy","medium","hard"])
    if st.button("🎲 Generate Quiz"):
        with st.spinner("Generating…"):
            resp = api_post("/test-yourself", {
                "topic": topic_sel, "num_questions": n_qs,
                "difficulty": difficulty, "solve_result": solve_result,
            })
        if resp and resp.get("success"):
            st.session_state["quiz_questions"] = resp["data"]["questions"]
            st.session_state["quiz_submitted"]  = False
            st.session_state["quiz_start"]      = time.time()

    questions = st.session_state.get("quiz_questions", [])
    if questions and not st.session_state.get("quiz_submitted", False):
        with st.form("quiz_form"):
            answers = {}
            for q in questions:
                opts = [o.split(") ",1)[1] if ") " in o else o for o in q["options"]]
                answers[q["id"]] = st.radio(
                    f"**Q{q['id']}: {q['question']}**", opts, key=f"q{q['id']}")
            submitted = st.form_submit_button("✅ Submit Answers")
        if submitted:
            score = 0
            duration = round(time.time()-st.session_state.get("quiz_start",time.time()), 2)
            st.session_state["quiz_submitted"] = True
            st.markdown("### 📊 Results")
            for q in questions:
                user_ans = answers[q["id"]]
                correct  = q["correct_answer"]
                if user_ans.strip() == correct.strip():
                    score += 1
                    st.success(f"✅ Q{q['id']}: Correct!")
                else:
                    st.error(f"❌ Q{q['id']}: Wrong – correct: **{correct}**")
            pct = round(score/len(questions)*100)
            c1, c2 = st.columns(2)
            c1.metric("Score", f"{score}/{len(questions)} ({pct}%)")
            c2.metric("Time",  f"{duration}s")
    if not questions:
        st.markdown("---")
        st.subheader("Quick Static Quiz")
        with st.form("quiz_static"):
            q1 = st.radio("Order of d²y/dx² + y = 0?", ["1","2","3"])
            q2 = st.radio("Is dy/dx = y separable?", ["Yes","No"])
            if st.form_submit_button("Submit"):
                score = (1 if q1=="2" else 0)+(1 if q2=="Yes" else 0)
                st.success(f"Score: {score}/2")

# =========================
# AI Resources
# =========================
elif page == "AI Resources":
    st.title("💬 AI Explanation & Resources")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    c1, c2 = st.columns([4,1])
    with c1:
        user_input = st.text_input("Ask about any ODE topic",
                                    placeholder="bernoulli, separation, exact…")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear"):
            st.session_state["chat_history"] = []
    if st.button("📤 Send") and user_input.strip():
        st.session_state["chat_history"].append(("User", user_input))
        resp = api_post("/explain", {
            "topic": user_input.strip(), "level":"beginner", "language":"arabic"})
        if resp and resp.get("success") and "error" not in resp["data"]:
            d = resp["data"]
            ai_msg = (f"**{d['topic'].title()}**\n\n{d['explanation']}\n\n"
                      f"📐 Formula: `{d['formula']}`\n\n🎥 [Watch Video]({d['video']})")
        else:
            ai_msg = (f"لم أجد شرحاً لـ '{user_input}'.\n"
                      f"جرّب: separation | linear | bernoulli | exact | homogeneous")
        st.session_state["chat_history"].append(("AI", ai_msg))
    for speaker, msg in reversed(st.session_state["chat_history"]):
        color = "#38bdf8" if speaker=="AI" else "#facc15"
        st.markdown(f'<div class="chat" style="color:{color};"><b>{speaker}:</b> {msg}</div>',
                    unsafe_allow_html=True)
    st.markdown("""
    <div class="card"><h4>📖 Learning Resources</h4><ul>
    <li><a href="https://www.khanacademy.org/math/differential-equations"
    target="_blank" style="color:#38bdf8;">Khan Academy ODE</a></li>
    <li><a href="https://tutorial.math.lamar.edu/classes/de/de.aspx"
    target="_blank" style="color:#38bdf8;">Paul's Online Notes</a></li>
    <li><a href="https://www.youtube.com" target="_blank"
    style="color:#38bdf8;">Tutorial Videos</a></li>
    </ul></div>""", unsafe_allow_html=True)