'''
Sub module for the churn library containing constants

Author: Patrick
Date: Oct 2022
'''
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

DATA_PATH = './data/bank_data.csv'
#Paths for the EDA plots
GENDER_PLT_PTH = './images/eda/gender.png'
AGE_PLT_PTH = './images/eda/age.png'
CORR_PLT_PTH = './images/eda/correlation.png'

#Paths for the classification report plots
CLS_REPORT_PLT_PTH = './images/results/classification_report.png'
ROC_CURVE_PLT_PTH = './images/results/roc_curve.png'
FEATURE_IMPORTANCE_PLT_LR_PTH = './images/results/feature_importances_lr.png'
FEATURE_IMPORTANCE_SHAP_PLT_LR_PTH = './images/results/feature_improtances_shap_lr.png'
FEATURE_IMPORTANCE_PLT_RFC_PTH = './images/results/feature_importances_rfc.png'
FEATURE_IMPORTANCE_SHAP_PLT_RFC_PTH = './images/results/feature_improtances_shap_rfc.png'

#Paths for the models
RFC_MODEL_PTH = './models/rfc_model.pkl'
LR_MODEL_PTH = './models/logistic_model.pkl'


# Columns to keep for feature engineering
KEEP_COLS = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    