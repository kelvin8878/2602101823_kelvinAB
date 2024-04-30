import streamlit as st
import joblib
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder  # For categorical variable encoding

# Load the machine learning model
model = joblib.load('XGBoost.pkl')

def main():
    st.title('Churn Model Deployment')

    credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
    age = st.number_input('Age', min_value=17, max_value=100)
    balance = st.number_input('Balance', min_value=0.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

    geography = st.selectbox('Geography', ['Spain', 'France', 'Germany'])  
    gender = st.selectbox('Gender', ['Male', 'Female'])  
    tenure = st.number_input('Tenure', min_value=0, max_value=10)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
    has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
    is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])  

    if st.button('Make Prediction'):
        # Encode categorical variables
        label_encoder = LabelEncoder()
        geography_encoded = label_encoder.fit_transform([geography])[0]
        gender_encoded = label_encoder.fit_transform([gender])[0]
        has_cr_card_encoded = label_encoder.fit_transform([has_cr_card])[0]
        is_active_member_encoded = label_encoder.fit_transform([is_active_member])[0]

        # Create the input features array in the correct order
        features = [
            credit_score, geography_encoded, gender_encoded, age, tenure, balance, 
            num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary
        ]


        result = make_prediction(features)
            
        # Modify output based on prediction (assuming 0 is negative, 1 is positive)
        if result == 0:
            output_text = "No"
        else:
            output_text = "Yes"

        st.success(f'The prediction is: {output_text}')

def make_prediction(features):
    # Reshape the input to a 2D array for the model
    input_array = np.array(features).reshape(1, -1) 
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
