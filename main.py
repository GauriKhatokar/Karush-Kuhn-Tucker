import streamlit as st
import sympy as sp
import numpy as np
from scipy.optimize import minimize

# -------------------------------
# Utility
# -------------------------------
def sanitise(expr: str):
    return expr.replace("^", "**")

# -------------------------------
# KKT Solver (UNCHANGED LOGIC)
# -------------------------------
def solve_kkt(obj_expr, constraints_list, obj_type, x0):
    n = len(x0)

    var_syms = sp.symbols(" ".join(f"x{i+1}" for i in range(n)))
    if n == 1:
        var_syms = (var_syms,)

    obj_expr = sanitise(obj_expr)

    parsed_constraints = []
    for expr in constraints_list:
        expr = sanitise(expr)

        if "<=" in expr:
            lhs, rhs = expr.split("<=")
            g = sp.sympify(rhs) - sp.sympify(lhs)

        elif ">=" in expr:
            lhs, rhs = expr.split(">=")
            g = sp.sympify(lhs) - sp.sympify(rhs)

        elif "=" in expr:
            lhs, rhs = expr.split("=")
            h = sp.sympify(lhs) - sp.sympify(rhs)
            parsed_constraints.append(("eq", h))
            continue

        parsed_constraints.append(("ineq", g))

    f_sym = sp.sympify(obj_expr)
    sign = -1 if obj_type.lower() == "maximize" else 1

    f_lam = sp.lambdify(var_syms, sign * f_sym, "numpy")

    def objective(x):
        return float(f_lam(*x))

    scipy_constraints = []
    for typ, expr in parsed_constraints:
        fn = sp.lambdify(var_syms, expr, "numpy")

        if typ == "ineq":
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda x, fn=fn: float(fn(*x))
            })
        else:
            scipy_constraints.append({
                "type": "eq",
                "fun": lambda x, fn=fn: float(fn(*x))
            })

    best_res = None
    starts = [np.array(x0, dtype=float)]

    np.random.seed(42)
    for _ in range(30):
        starts.append(np.random.uniform(0, 5, n))

    for x_init in starts:
        try:
            res = minimize(objective, x_init, method="SLSQP", constraints=scipy_constraints)

            if res.success:
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
        except:
            continue

    if best_res is None:
        raise RuntimeError("Solver failed.")

    x_star = best_res.x
    f_star = float(f_lam(*x_star)) * sign

    return x_star, f_star, []


# -------------------------------
# 🎨 PREMIUM MATH BACKGROUND UI
# -------------------------------
import streamlit as st
import base64

st.set_page_config(page_title="KKT Solver", layout="centered")

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Dark overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 23, 42, 0.85);
        z-index: 0;
    }}

    .block-container {{
        position: relative;
        z-index: 1;
    }}

    input, textarea {{
        background-color: #020617 !important;
        color: white !important;
        border-radius: 8px !important;
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #6366f1, #22d3ee);
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
        font-weight: bold;
    }}

    </style>
    """, unsafe_allow_html=True)

set_bg("bg.png")
# -------------------------------
# HEADER
# -------------------------------
st.markdown("# 📊 KKT Solver")
st.markdown("### Karush–Kuhn–Tucker Optimization Tool")

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("⚙️ Problem Setup")

obj_type = st.radio("Objective Type", ["Maximize", "Minimize"])

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    obj_expr = st.text_input(
        "Objective Function",
        "8*x1 + 10*x2 - x1^2 - x2^2"
    )

    constraints_input = st.text_area(
        "Constraints",
        "3*x1 + 2*x2 <= 6\nx1 >= 0\nx2 >= 0"
    )

with col2:
    x0_input = st.text_input("Initial Guess", "0,0")

# -------------------------------
# BUTTON
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Solve Optimization Problem"):

    try:
        constraints_list = [line.strip() for line in constraints_input.split("\n") if line.strip()]
        x0 = [float(v.strip()) for v in x0_input.split(",")]

        x_star, f_star, _ = solve_kkt(obj_expr, constraints_list, obj_type, x0)

        st.divider()
        st.subheader("📈 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"x1 ≈ {x_star[0]:.4f}")
            st.success(f"x2 ≈ {x_star[1]:.4f}")

        with col2:
            st.info(f"Z ≈ {f_star:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
# Ctrl + C
# python -m streamlit run main.py