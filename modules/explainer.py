data = {
    "separation": {
        "explanation": "Separation of variables is a method used to solve differential "
                       "equations by separating variables on each side.",
        "formula": "dy/dx = g(x)h(y) → ∫(1/h(y)) dy = ∫g(x) dx",
        "video": "https://youtu.be/oAi2QWhFZX4?si=UPdZCKKcWGobOJe5"
    },
    "homogeneous": {
        "explanation": "A homogeneous differential equation has functions that can be "
                       "expressed in terms of y/x.",
        "formula": "dy/dx = F(y/x)",
        "video": "https://youtu.be/oAi2QWhFZX4?si=UPdZCKKcWGobOJe5"
    },
    "non homogeneous": {
        "explanation": "A non-homogeneous differential equation includes an extra function "
                       "or term not depending only on the variables.",
        "formula": "dy/dx + P(x)y = Q(x)",
        "video": "https://youtu.be/K_ZWLFE5Enw?si=W1n5ID8WL0qCy4VK"
    },
    "linear": {
        "explanation": "A linear differential equation is one where the function and its "
                       "derivatives appear linearly.",
        "formula": "dy/dx + P(x)y = Q(x)",
        "video": "https://youtu.be/iiLoCO1unEs?si=_CG4KOB1M6mDhYK6"
    },
    "bernoulli": {
        "explanation": "A Bernoulli differential equation is a nonlinear equation that can be "
                       "transformed into a linear one using substitution.",
        "formula": "dy/dx + P(x)y = Q(x)y^n",
        "video": "https://youtu.be/iiLoCO1unEs?si=_CG4KOB1M6mDhYK6"
    },
    "exact": {
        "explanation": "An exact differential equation satisfies the condition Mdx + Ndy = 0 "
                       "where ∂M/∂y = ∂N/∂x.",
        "formula": "M(x,y)dx + N(x,y)dy = 0",
        "video": "https://youtu.be/HwBvnJxvlFA?si=wCDQhhmYFMb2v7WV"
    },
    "non exact": {
        "explanation": "A non-exact differential equation does not satisfy the exact condition "
                       "and may require integrating factor.",
        "formula": "M_y ≠ N_x",
        "video": "https://youtu.be/OxMgw6UiNJ8?si=UZM9EKE2hZrbBQ47"
    }
}


def explain_topic(topic: str, level: str = "beginner", language: str = "arabic") -> dict:
    """Called by app.py /explain endpoint."""
    key = topic.strip().lower()
    if key in data:
        return {
            "topic": topic,
            "explanation": data[key]["explanation"],
            "formula":     data[key]["formula"],
            "video":       data[key]["video"],
            "level":       level,
            "language":    language,
        }
    return {
        "topic": topic,
        "error": f"Topic '{topic}' not found.",
        "available_topics": list(data.keys())
    }


def list_topics() -> list:
    return list(data.keys())