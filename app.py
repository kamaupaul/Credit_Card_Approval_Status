import streamlit as st
import shap
from predictor import main


page_bg_img = """
<style>
[data-testid="stSidebar"] {
    background-color:  rgba(107, 142, 168, 0.8);
    background-size: cover;
}
</style>
"""
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] {
        color: purple;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(page_bg_img, unsafe_allow_html=True)

main()
