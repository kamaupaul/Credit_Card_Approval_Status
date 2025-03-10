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


def page_home():
    
    page_bg_img = """
<style>
[class="appview-container css-1wrcr25 e1g8pov66"] {
background-image: url("https://www.investopedia.com/thmb/pxxo1p6473mB4XmnwmSeAUdunSQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1203763961-880a467befb64f39856f3fc7904ae12a.jpg");
background-size: cover;
opacity: 1.0;
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
    unsafe_allow_html=True)

    st.markdown(page_bg_img, unsafe_allow_html= True)

    st.markdown('<h1 style="color: purple;">Metric Marvin Credit Card Approver App</h1>',unsafe_allow_html=True)
    st.markdown("""
                <div style="background-color: rgba(173, 216, 230, 0.7); padding: 20px;">
                <h3><em style="color: purple;">your credit card approver co-pilot</em></h3>
                <p><h6>Unleash the Power of Credit Card Approval
            
            Welcome to the Credit Card Approval App! Get ready for an exciting journey as we revolutionize credit card approvals.
            
            Our cutting-edge platform uses advanced algorithms and machine learning to provide valuable insights to be used in targeted marketing and as co-pilot to your existing system.
            
            Join us today and experience streamlined credit card approvals. Achieve your financial goals and enjoy the lifestyle you deserve!<h6></p>
                </div>
                """,
        unsafe_allow_html=True
        )
    
    
def page_info():
    page_bg_img1 = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.prismic.io/axerve/94020ecc-3c87-4a2a-bfed-56402a421ad8_Carta%20di%20credito%20-%20Blog.png?ixlib=gatsbyFP&auto=compress%2Cformat&fit=max");
background-size: cover;
opacity: 0.9;
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
    st.markdown(page_bg_img1, unsafe_allow_html= True)
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
    age_years = st.sidebar.slider('AGE',18,100,18)
    Employment_Duration= st.sidebar.slider('Years of Experience',0,100,0)
    x =  np.array([[Gender,Own_Car,Own_Property,income_type,income_range,Family_Status,Housing_Type,Employment_Duration,Occupation,Education_level,Num_Family,age_years,]])
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

    decision_tree = best_estimator.named_steps['tree']
    explainer = shap.TreeExplainer(decision_tree)
    shap_values = explainer.shap_values(df)
    column_names = df.columns.tolist()
    shap_values_reshaped = np.array(shap_values).reshape(2, 12)  # Convert to numpy array and reshape
    shap_df = pd.DataFrame(shap_values_reshaped, columns=column_names)
    shap_values_std = np.std(shap_values_reshaped, axis=0)
    shap_std_df = pd.DataFrame({'Feature': column_names, 'importance': shap_values_std})
    explanations = []
    for index, row in shap_std_df.iterrows():
        feature = row['Feature']
        importance = row['importance']
        if importance < 0.05:
            recommendation = f"{feature} value is a bit low, this contributes to probable denial."
        else:
            recommendation = f"{feature} value is fairly high will add to a possible approval."
        explanations.append({'Feature': feature, 'importance': importance, 'Explanation': recommendation})
    explanations_df = pd.DataFrame(explanations)
    pd.set_option('display.max_colwidth', None)
    shap.summary_plot(shap_values_reshaped, column_names, plot_type='bar')
    # Get the feature importance values from SHAP values
    shap_values_std = np.std(shap_values_reshaped, axis=0)
    feature_importance = pd.DataFrame({'Feature': column_names, 'Importance': shap_values_std})

    # Sort the feature importance in descending order
    feature_importance = feature_importance.sort_values(by='Importance', ascending=True)
    mortgage_threshold = 0.04
    car_loan_threshold = 0.02
    insurance_threshold = 0.02
    money_market = 0.01

    # Check the importance values and make recommendations
    if shap_std_df.loc[shap_std_df['Feature'] == 'Income_Range', 'importance'].values[0] > mortgage_threshold and \
            shap_std_df.loc[shap_std_df['Feature'] == 'Own_Property', 'importance'].values[0] < car_loan_threshold and \
                df['Own_Property'].values[0] == 0:
        recommendation = "Recommend a mortgage package."
    elif shap_std_df.loc[shap_std_df['Feature'] == 'Income_Type', 'importance'].values[0] < 0.009 and \
           shap_std_df.loc[shap_std_df['Feature'] == 'Income_Range', 'importance'].values[0] < 0.09 and \
           df['Own_Property'].values[0] == 1 and\
               df['Own_Car'].values[0] == 1 :
        recommendation = "Recommend a money market investment fund."
    elif shap_std_df.loc[shap_std_df['Feature'] == 'Own_Car', 'importance'].values[0] > car_loan_threshold and \
            df['Own_Car'].values[0] == 0 and \
            shap_std_df.loc[shap_std_df['Feature'] == 'Income_Range', 'importance'].values[0] > 0.02:
        recommendation = "Recommend a car loan instead."
    elif shap_std_df.loc[shap_std_df['Feature'] == 'Num_Family', 'importance'].values[0] > 0.01:
        recommendation = "Recommend an education or life insurance policy."
    elif shap_std_df.loc[shap_std_df['Feature'] == 'Income_Type', 'importance'].values[0] < 0.002 and \
            shap_std_df.loc[shap_std_df['Feature'] == 'Income_Range', 'importance'].values[0] > 0.05:
        recommendation = "Recommend a student loan."
    else:
        recommendation = "No specific recommendation."

    # Set the background color and border radius based on the recommendation
    background_color = "rgba(107, 142, 168, 0.8)" if recommendation != "No specific recommendation" else "rgba(255, 0, 0, 0.4)"
    border_radius = "0px"
    status_color = "blue" if recommendation != "No specific recommendation" else "red"

    # Create the output text with the recommendation in a text box
    output_text = f"<div style='background-color: {background_color}; padding: 20px; border-radius: {border_radius};'>"
    output_text += f"<h3 style='color: {status_color};'>{recommendation}</h3>"
    output_text += "</div>"

    # Display the recommendation in a text box
    
    # Display the recommendation
    

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(11, 6))

    # Create horizontal bars and color code based on shap_std values
    bars = ax.bar(feature_importance['Feature'], feature_importance['Importance'],
                color=np.where(feature_importance['Importance'] < 0.05, 'r', 'b'))

    # Add values on the right side of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 3),
                ha='center', va='bottom')

    ax.set_ylabel('Importance')
    ax.set_xlabel('Feature')
    ax.set_title('Feature Importance')
    # Rotate x-axis labels for better visibility if needed


    # Display the styled DataFrame
    st.dataframe(explanations_df)
    plt.xticks(rotation=90)
    #st.write(explanations_df)
    st.empty()
    st.pyplot(fig)
    st.markdown(output_text, unsafe_allow_html=True)
    #st.pyplot(fig)
    #st.set_option('deprecation.showPyplotGlobalUse', False)

page_bg_img1 = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://assets-global.website-files.com/6038c1030be2580db46ccf46/619661368989585e94fda3eb_Happy-credit-card-user-pros-and-cons%20(1).jpg");
background-size: cover;
opacity: 0.9;
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
       
def show_predict_page():
    st.markdown(page_bg_img1, unsafe_allow_html= True)


     
best_estimator = final_model.best_estimator_
def predict_approval(x):
    # Create a DataFrame with feature names
    columns = ['Gender', 'Own_Car', 'Own_Property', 'Income_Type', 'Income_Range',
            'Family_Status', 'Housing_Type', 'Employment_Duration', 'Occupation',
            #'Education_Level',
              'Num_Family', 'age(years)']
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
    #df['Education_Level'] = le_Education_Level.transform(df['Education_Level'])
    result = best_estimator.predict(df)
    if result[0] == 1:
        approval_status = 'Approved'
        status_color = "blue"
    else:
        approval_status = 'Denied'
        status_color = "black"

    # Display the approval status and reason
    # Display the output in a text box with colored text
    output_text = f"<div style='background-color: rgba(107, 142, 168, 0.8); padding: 20px; border-radius: 0px;'>"
    if approval_status == 'Denied':
        output_text = f"<div style='background-color: rgba(255, 0, 0, 0.4); padding: 20px; border-radius: 0px;'>"

    output_text += f"<h1 style='color: {status_color};'>{approval_status}</h1>"
    output_text += "</div>"

    st.markdown(output_text, unsafe_allow_html=True)
    if st.button('clear'):
         page_info()
        

        
    # Make the prediction



def user_input_features():
    Own_Property = ('Y','N')
    Own_Car = ('Y','N')
    Gender = ('F','M')
    st.sidebar.markdown("<div class='sidebar-title'>input client details</div>", unsafe_allow_html=True)
    # Add a text box for client name
    st.sidebar.text_input('Client Name')
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
    age_years = st.sidebar.slider('AGE',18,100,18)
    Employment_Duration= st.sidebar.slider('Years of Experience',0,100,0)

    ok = st.button('Get Status')
    if ok:
        x =  np.array([[Gender,Own_Car,Own_Property,income_type,income_range,Family_Status,Housing_Type,Employment_Duration,Occupation,#Education_level,
                        Num_Family,age_years,]])
    # Call the prediction function
        predict_approval(x)
        

        
        
def main():
    # Set the title and description
    #st.markdown("<h5 style='text-align: justified; color: black;'>NAVIGATION</h1>", unsafe_allow_html=True) 
    with st.container():
        pages = ["Home", "Approver", "Analyze"]
        selected_page = st.selectbox("Go to", pages)

        
    if selected_page == "Home":
        page_home()
           
    elif selected_page == "Approver":
        show_predict_page()
        user_input_features()

    elif selected_page == "Analyze":
        page_info()
    


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

if __name__ == "__main__":
    main()

