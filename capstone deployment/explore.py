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
le_Housing_Type =data['le_Housing_Type']
le_Occupation =data['le_Occupation']
le_Education_Level =data['le_Education_Level']



def show_predict_page():

    st.title(" Bank Credit Card Approver")
    st.write("""### your credit card approver co-pilot""")


def predict_approval(x):
    # Create a DataFrame with feature names
    columns = ['Gender', 'Own_Car', 'Own_Property', 'Income_Type', 'Income_Range',
               'Family_Status', 'Housing_Type', 'Employment_Duration', 'Occupation',
               'Education_Level', 'Num_Family', 'age(years)']
    df = pd.DataFrame(x, columns=columns)

    # Transform categorical variables using label encoders
    df['Gender'] = le_gender.transform(df['Gender'])
    df['Own_Car'] = le_car.transform(df['Own_Car'])
    df['Own_Property'] = le_property.transform(df['Own_Property'])
    df['Income_Type'] = le_Income_Type.transform(df['Income_Type'])
    df['Income_Range'] = le_Income_Range.transform(df['Income_Range'])
    df['Family_Status'] = le_Family_Status.transform(df['Family_Status'])
    df['Housing_Type'] = le_Housing_Type.transform(df['Housing_Type'])
    df['Occupation'] = le_Occupation.transform(df['Occupation'])
    df['Education_Level'] = le_Education_Level.transform(df['Education_Level'])
    # Make the prediction
    best_estimator = final_model.best_estimator_
    decision_tree = best_estimator.named_steps['tree']
    explainer = shap.TreeExplainer(decision_tree)
    shap_values = explainer.shap_values(df)
    column_names = df.columns.tolist()
    shap_values_reshaped = np.array(shap_values).reshape(2, 12)  # Convert to numpy array and reshape
    shap_df = pd.DataFrame(shap_values_reshaped, columns=column_names)
    shap_values_std = np.std(shap_values_reshaped, axis=0)
    shap_std_df = pd.DataFrame({'Feature': column_names, 'Std': shap_values_std})
    plot = shap.summary_plot(shap_values_reshaped, column_names, plot_type='bar')
    result = best_estimator.predict(df)
    

    if result[0] == 1:
        approval_status = 'Approved'
        reason = 'Reason:'
    else:
        approval_status = 'Denied'
        reason = 'Reason:'
        

    
    # Display the approval status and reason
    st.subheader(approval_status)
    st.write(reason, shap_std_df)
    return plot

         
def user_input_features():

    Own_Property = ('Y','N')
    Own_Car = ('Y','N')
    Gender = ('F','M')
    Gender = st.sidebar.selectbox('Gender',Gender)
    Family_Status =('Married','Single','Civil marriage','Separated','Widow')
    Family_Status=st.sidebar.selectbox('Marital Status',Family_Status)
    Housing_Type = ('House / apartment',
                   'With parents',
                   'Municipal apartment',
                   'Rented apartment',
                   'Office apartment',
                   'Co-op apartment',)
    Housing_Type=st.sidebar.selectbox('Type of House',Housing_Type)
    Occupations=('unspecified',
                 'Laborers',
                 'Core staff',
                 'Sales staff',
                 'Managers',
                 'Drivers',
                 'High skill tech staff',
                 'Accountants',
                 'Medicine staff',
                 'Cooking staff',
                 'Security staff',
                 'Cleaning staff',
                 'Private service staff',
                 'Low-skill Laborers',
                 'Waiters/barmen staff',
                 'Secretaries',
                 'HR staff',
                 'Realty agents',
                 'IT staff',)
    Education_level = ('Secondary','Higher education','Incomplete higher','Lower secondary','Academic degree')
    income_range = ('27000-54000',
'54000-81000','81000-108000','108000-135000','135000-162000','162000-189000','189000-216000','216000-243000','243000-270000','270000-297000',
'297000-324000','324000-351000','351000-378000','378000-405000','405000-432000','432000-459000','459000-486000','486000-513000',
'513000-540000','540000-567000','567000-594000','594000-621000','621000-648000','648000-675000','675000-702000','702000-729000',
'729000-756000','756000-783000','783000-810000','810000-837000','891000-918000','945000-972000','972000-999000','1107000-1134000',
'1350000-1377000','1566000-1593000')
    income_type = ('Working','Commercial associate','Pensioner','State servant','Student')
    Occupation=st.sidebar.selectbox('Occupation',Occupations)
    Own_Car=st.sidebar.selectbox('Car owner',Own_Car)
    Own_Property=st.sidebar.selectbox('Property',Own_Property)
    Education_level=st.sidebar.selectbox('Education',Education_level)
    income_type=st.sidebar.selectbox('Income type',income_type)
    income_range=st.sidebar.selectbox('Income range',income_range)
    Num_Family = st.sidebar.slider('Number of Family members',0,20,0)
    age_years = st.sidebar.slider('AGE',0,100,0)
    Employment_Duration= st.sidebar.slider('Years of Experience',0,100,0)
    ok = st.button('Get Status')
    if ok:
        x =  np.array([[Gender,Own_Car,Own_Property,income_type,income_range,Family_Status,Housing_Type,Employment_Duration,Occupation,Education_level,Num_Family,age_years,]])
        # Call the prediction function
        predict_approval(x)

if __name__ == '__main__':
    show_predict_page()
    user_input_features()

         
  
        
