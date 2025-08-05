import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier


model = joblib.load('models/final_xgb_model.pkl')


st.set_page_config(page_title="E-Commerce Churn Prediction", layout="centered")
st.title("E-Commerce Churn Prediction")
st.markdown("Fill out customer details to check if they'll churn.")

with st.form("input form"):
    tenure = st.number_input("Tenure (in days)", min_value=0)
    warehouse_to_home = st.number_input("Warehouse to Home Distance",min_value=0)
    satisfaction_score = st.slider("Satisfactory Score",1,5)
    num_devices = st.number_input("Number of Devices Registered",min_value=0)
    order_cat = st.selectbox("Preferred Order Category", ['Laptop & Accessory', 'Mobile', 'Others', 'Fashion', 'Grocery'])
    marital = st.selectbox("Marital Status",['Single','Married','Divorced'])
    num_address = st.number_input("Number of Addresses",min_value=0)
    complain = st.selectbox("Customer Complaint",['Yes','No'])
    days_since_order = st.number_input("Days since Last Order",min_value=0)
    cashback = st.number_input("Casback Received",min_value=0.0)

    submitted = st.form_submit_button("Predict Churn")

def encode_inputs(preferred_order_cart, marital_status):
    enc = {
        'PreferedOrderCat_Fashion': 0,
        'PreferedOrderCat_Grocery': 0,
        'PreferedOrderCat_Laptop & Accessory': 0,
        'PreferedOrderCat_Mobile': 0,
        'PreferedOrderCat_Others': 0,
        'MaritalStatus_Divorced': 0,
        'MaritalStatus_Married': 0,
        'MaritalStatus_Single': 0
    }

    enc[f'PreferedOrderCat_{preferred_order_cart}'] = 1
    enc[f'MaritalStatus_{marital_status}'] = 1
    return enc

base = {
    'Tenure': int(tenure),
    'WarehouseToHome': int(warehouse_to_home),
    'NumberOfDeviceRegistered': int(num_devices),
    'SatisfactionScore':int(satisfaction_score),
    'NumberOfAddress': int(num_address),
    'Complain': 1 if complain=='Yes' else 0,
    'DaySinceLastOrder': int(days_since_order),
    'CashbackAmount': float(cashback)
}


if submitted:
    input_data = {**base, **encode_inputs(order_cat,marital)}
    df_input = pd.DataFrame([input_data])
    df_input = df_input.astype({
        'Tenure': 'int',
        'WarehouseToHome': 'int',
        'NumberOfDeviceRegistered': 'int',
        'SatisfactionScore': 'int',
        'NumberOfAddress': 'int',
        'Complain': 'int',
        'DaySinceLastOrder': 'int',
        'CashbackAmount': 'float'
    })

    df_input['IsNewCustomer'] = (df_input['Tenure'] <= 5).astype('int')
    df_input['LowSatisfaction'] = (df_input['SatisfactionScore'] <= 2).astype('int')
    df_input['HighCashback'] = (df_input['CashbackAmount'] > df_input['CashbackAmount'].median()).astype('int')
    df_input['RecentlyActive'] = (df_input['DaySinceLastOrder'] <= 3).astype('int')

    model_features = model.get_booster().feature_names
    df_input = df_input[model_features]

    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]
    threshold = 0.6
    prediction = 1 if proba >= threshold else 0


    st.header("Prediction Result") 
    st.write("Churn Probabilty: ",round(proba * 100, 2), "%")

    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("The Customer is likely to Stay")

