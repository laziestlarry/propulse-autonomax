
import streamlit as st
import subprocess
import os

st.set_page_config(page_title="AutonomaX OPS Panel", layout="centered")
st.title("🧠 AutonomaX Tactical Engine")

st.markdown("🚀 Tactical control center for executing automation strategies.")
port = os.getenv("PORT", "8080")  # Fallback to 8080 if not provided

if st.button("🔥 Run All Tactics Now"):
    st.info("Executing tactics...")
    result = subprocess.run(["python3", "autonomax_tactical_engine.py"], capture_output=True, text=True)
    st.code(result.stdout + result.stderr)
    st.success("Tactics executed.")

st.markdown(f"🛰️ Streamlit is serving on port `{port}` (auto-detected)")
st.markdown("⏱️ Background refresh interval: 15 minutes")
