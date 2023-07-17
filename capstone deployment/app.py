import streamlit as st
import shap
from predictor import show_predict_page, user_input_features

show_predict_page()
page_bg_img = """
<style>
[data-testid="stSidebar"] {
background-image: url("https://img.freepik.com/free-vector/realistic-credit-card-design_23-2149126088.jpg?w=2000");
background-size: cover;
}
</style>
"""
st.markdown(
    """
    <style>
    [data-testid="stMarkdownContainer"] {
        color: purple;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(page_bg_img, unsafe_allow_html= True)
user_input_features()
