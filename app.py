import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="KKT Solver", layout="centered")

st.title("📊 KKT Solver")

# ---------------- BACKGROUND  ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

set_bg("bg.png")





# ---------------- INPUT ----------------
objective_type = st.radio("Objective", ["Minimize", "Maximize"])
view_mode = st.radio(
    "View Mode",
    ["📊 Full Solution (Step-by-Step)", "⚡ Quick Result Only"]
)

obj_input = st.text_input("Objective Function (Z)", "x**2 + y**2")
constraint_input = st.text_input("Constraint (use <=, >=, =)", "x + y >= 2")
# ---------------- PARSE ----------------
def parse_constraint(expr):
    if ">=" in expr:
        left, right = expr.split(">=")
        return sp.sympify(right) - sp.sympify(left)
    elif "<=" in expr:
        left, right = expr.split("<=")
        return sp.sympify(left) - sp.sympify(right)
    elif "=" in expr:
        left, right = expr.split("=")
        return sp.sympify(left) - sp.sympify(right)
# ---------------- MAIN ----------------
if st.button("🚀 Solve"):

    try:
        f = sp.sympify(obj_input)
        g = parse_constraint(constraint_input)

        vars = sorted(list(f.free_symbols.union(g.free_symbols)), key=lambda x: str(x))
        lamb = sp.symbols('lambda', real=True)

        # Maximize → convert to minimize
        if objective_type == "Maximize":
            f = -f

        # ---------------- LAGRANGIAN ----------------
         L = f - lamb * g

        # ---------------- STATIONARITY ----------------


        # ---------------- FULL SOLUTION ----------------


        # ---------------- RESULT ----------------


        # ---------------- QUICK MODE ----------------


        # ---------------- GRAPH ----------------
