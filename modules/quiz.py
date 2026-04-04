# Module 5 - Quiz Engine (ODEzilla)
import random

# ── Rich question bank per ODE type ─────────────────────────────────────────
# FIX 3: questions are now tied to the actual ODE type & method the solver returned
_QUESTION_BANK = {
    "separation": [
        {"text": "What is the main idea of separation of variables?",
         "correct": "Separate x terms on one side and y terms on the other",
         "distractors": ["Use an integrating factor", "Find the characteristic equation",
                         "Apply undetermined coefficients"]},
        {"text": "Which form is required for separation of variables?",
         "correct": "dy/dx = g(x) · h(y)",
         "distractors": ["dy/dx + P(x)y = Q(x)", "M dx + N dy = 0", "dy/dx = F(y/x)"]},
    ],
    "linear": [
        {"text": "What is the standard form of a first-order linear ODE?",
         "correct": "dy/dx + P(x)y = Q(x)",
         "distractors": ["dy/dx = g(x)h(y)", "M dx + N dy = 0", "dy/dx = F(y/x)"]},
        {"text": "Which technique solves first-order linear ODEs?",
         "correct": "Integrating factor",
         "distractors": ["Undetermined coefficients", "Separation of variables",
                         "Characteristic equation"]},
    ],
    "bernoulli": [
        {"text": "Which substitution linearises a Bernoulli equation?",
         "correct": "v = y^(1-n)",
         "distractors": ["v = y/x", "v = e^y", "v = ln(y)"]},
        {"text": "What is the form of a Bernoulli ODE?",
         "correct": "dy/dx + P(x)y = Q(x)y^n",
         "distractors": ["dy/dx + P(x)y = Q(x)", "M dx + N dy = 0",
                         "dy/dx = g(x)h(y)"]},
    ],
    "exact": [
        {"text": "What condition makes an ODE exact?",
         "correct": "∂M/∂y = ∂N/∂x",
         "distractors": ["M = N", "∂M/∂x = ∂N/∂y", "M · N = 1"]},
        {"text": "What is the standard form of an exact ODE?",
         "correct": "M(x,y)dx + N(x,y)dy = 0",
         "distractors": ["dy/dx + P(x)y = Q(x)", "dy/dx = g(x)h(y)",
                         "dy/dx = F(y/x)"]},
    ],
    "homogeneous": [
        {"text": "What substitution is used for homogeneous ODEs?",
         "correct": "v = y/x  →  y = vx",
         "distractors": ["v = y^(1-n)", "v = e^y", "v = x·y"]},
        {"text": "What form identifies a homogeneous ODE?",
         "correct": "dy/dx = F(y/x)",
         "distractors": ["dy/dx + P(x)y = Q(x)", "M dx + N dy = 0",
                         "dy/dx = g(x)h(y)"]},
    ],
    "euler": [
        {"text": "What is the update formula in Euler's method?",
         "correct": "y_(n+1) = y_n + h · f(t_n, y_n)",
         "distractors": ["y_(n+1) = y_n + (h/6)(k1+2k2+2k3+k4)",
                         "y_(n+1) = y_n · e^(h·f)",
                         "y_(n+1) = y_n - h · f(t_n, y_n)"]},
        {"text": "What is the order of accuracy of Euler's method?",
         "correct": "First-order (O(h))",
         "distractors": ["Second-order (O(h²))", "Fourth-order (O(h⁴))", "Zero-order"]},
    ],
    "rk4": [
        {"text": "How many slope estimates does RK4 use per step?",
         "correct": "4 (k1, k2, k3, k4)",
         "distractors": ["2", "1", "6"]},
        {"text": "What is the order of accuracy of RK4?",
         "correct": "Fourth-order (O(h⁴))",
         "distractors": ["First-order (O(h))", "Second-order (O(h²))", "Third-order (O(h³))"]},
    ],
    # Physics models
    "population growth model": [
        {"text": "Which ODE models exponential population growth?",
         "correct": "dy/dt = k·y",
         "distractors": ["dy/dt = r·y(1 - y/K)", "dy/dt = -k(y - T_env)",
                         "dy/dt = g - (c/m)·y"]},
    ],
    "logistic growth model": [
        {"text": "What term limits growth in the logistic model?",
         "correct": "(1 - y/K) where K is the carrying capacity",
         "distractors": ["e^(-ky)", "1/y", "K - y"]},
    ],
    "cooling": [
        {"text": "Newton's Cooling law states that cooling rate is proportional to:",
         "correct": "The difference between object temperature and environment",
         "distractors": ["The object's temperature alone", "Time elapsed",
                         "The square of the temperature"]},
    ],
    "pendulum": [
        {"text": "What makes the simple pendulum ODE nonlinear?",
         "correct": "The sin(θ) term",
         "distractors": ["The angular velocity term", "The length L",
                         "Gravity constant g"]},
    ],
}

_GENERIC_DISTRACTORS = [
    "Separation of Variables", "Integrating Factor",
    "Undetermined Coefficients", "Characteristic Equation",
    "Partial Differential Equation", "Euler's Method", "RK4"
]


class QuizEngine:
    def __init__(self, solution_data=None):
        """
        solution_data: dict passed from the solver with keys:
            'type'   – ODE classification (e.g. "linear", "bernoulli")
            'method' – numerical method used (e.g. "RK4", "Euler")
            'model'  – physics model name if applicable (e.g. "Pendulum")
        If None, quiz falls back to topic-based generation.
        """
        self.solution_data = solution_data or {}

    # ── FIX 3: generate_quiz now uses the real solver output ─────────────────
    def generate_quiz(self):
        """
        Picks a question based on what the solver actually returned.
        Uses solution_data['type'] and solution_data['method'] to choose
        relevant questions from the bank.
        """
        ode_type  = (self.solution_data.get("type") or "").lower()
        method    = (self.solution_data.get("method") or "").lower().replace(" ", "").replace("(scipy)", "")
        model_title = (self.solution_data.get("model") or "").lower()

        # Build candidate pool from the bank
        bank = []
        for key in [ode_type, method, model_title]:
            if key in _QUESTION_BANK:
                bank.extend(_QUESTION_BANK[key])

        if bank:
            selected = random.choice(bank)
        else:
            # Fallback to generic classification question
            selected = {
                "text": f"What is the classification of this ODE?",
                "correct": self.solution_data.get("type", "Unknown"),
                "distractors": ["First-order Linear", "Partial Differential", "Non-linear Exact"]
            }

        options = selected["distractors"][:3] + [selected["correct"]]
        random.shuffle(options)
        return selected["text"], options, selected["correct"]

    def evaluate(self, user_answer, correct_answer):
        if user_answer.strip() == correct_answer.strip():
            return 100, "✔ Excellent! Correct answer."
        else:
            return 0, f"❌ Incorrect. The right answer is: {correct_answer}"

    # ── generate_from_topic: used by /test-yourself endpoint ─────────────────
    def generate_from_topic(self, topic: str, num_questions: int = 5,
                             difficulty: str = "medium",
                             solution_data: dict = None) -> list:
        """
        Generates MCQ list for the /test-yourself endpoint.
        If solution_data is provided (from /solve), questions are tailored
        to what the student actually solved — FIX 3.
        """
        # Merge solution_data into self so generate_quiz() uses it
        if solution_data:
            self.solution_data = solution_data
        else:
            self.solution_data = {"type": topic, "method": "RK4"}

        questions = []
        seen = set()
        attempts = 0
        while len(questions) < num_questions and attempts < num_questions * 5:
            attempts += 1
            text, options, correct = self.generate_quiz()
            if text not in seen:
                seen.add(text)
                questions.append({
                    "id":             len(questions) + 1,
                    "question":       text,
                    "options":        [f"{chr(0x61+j)}) {opt}" for j, opt in enumerate(options)],
                    "correct_answer": correct,
                    "difficulty":     difficulty,
                })
        return questions