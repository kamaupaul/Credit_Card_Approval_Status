import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import shap

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
final_model = data['model']
le_gender = data['le_gender']
le_car = data['le_car']
le_property = data['le_property']
le_Income_Type = data['le_Income_Type']
le_Income_Range = data['le_Income_Range']
le_Family_Status = data['le_Family_Status']
le_Housing_Type = data['le_Housing_Type']
le_Occupation = data['le_Occupation']
le_Education_Level = data['le_Education_Level']

def show_predict_page():
    st.title("Bank Credit Card Approver")
    st.write("""### Your credit card approver co-pilot""")
     
def predict_approval(x):
    # Rest of the code...

def user_input_features():
    # Rest of the code...
    ok = st.button('Get Status')
    if ok:
        x =  np.array([[Gender, Own_Car, Own_Property, income_type, income_range, Family_Status, Housing_Type, Employment_Duration, Occupation, Education_level, Num_Family, age_years]])
        predict_approval(x)
        show_predict_page()  # Add this line

# Call the user_input_features() function to start the app
user_input_features()
