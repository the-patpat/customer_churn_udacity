# library doc string
'''
This module contains functions to perform data analysis and machine learning on
the Kaggle Credit Card customers dataset to predict customers who are likely to churn.

Author: Patrick
Date: October 2022
'''

# import libraries
import os
import logging
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'
sns.set()



def import_data(input_pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    try:
        input_df = pd.read_csv(input_pth)
        input_df.head()
        return input_df
    except FileNotFoundError:
        logging.error("Could not read file %s", input_pth)

def perform_eda(input_df):
    '''
    perform eda on df and save figures to images folder
    prints out to console / log
    input:
            df: pandas dataframe

    output:
            None
    '''
    #Get the EDA attributes
    shape = input_df.shape
    null_count_per_column = input_df.isnull().sum()
    df_summary = input_df.describe()

    #Print out the attributes
    print(f"Dataframe has {shape[0]} rows and {shape[1]} columns.")
    print("The amount of rows with null values per column are as follows:")
    print(null_count_per_column)
    print("The summary of the dataframe is as follows:")
    print(df_summary)

    #Log the attributes
    logging.info("Dataframe has %d rows and %d columns.", shape[0], shape[1])
    logging.info("The amount of rows with null values per column are as follows:\n%s",
                null_count_per_column)
    logging.info("The summary of the dataframe is as follows:\n%s", df_summary)

    #Save the plots (this seems kind of repetitive)
    ##Univariate categorical
    plt.figure(figsize=(20,10))
    input_df['Gender'].hist()
    plt.savefig('./images/eda/gender.png')

    ##Univariate quantitative
    plt.figure(figsize=(20,10))
    input_df['Customer_Age'].hist()
    plt.savefig('./images/eda/age.png')

    ##Bivariate
    plt.figure(figsize=(20,10))
    sns.heatmap(input_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/correlation.png')

def encoder_helper(input_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    #Create Churn column: Only 0 if existing customer, else 1
    input_df['Churn'] = input_df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # gender encoded column
    gender_lst = []
    gender_groups = input_df.groupby('Gender').mean()['Churn']

    for val in input_df['Gender']:
        gender_lst.append(gender_groups.loc[val])

    input_df['Gender_Churn'] = gender_lst    
    #education encoded column
    edu_lst = []
    edu_groups = input_df.groupby('Education_Level').mean()['Churn']

    for val in input_df['Education_Level']:
        edu_lst.append(edu_groups.loc[val])

    input_df['Education_Level_Churn'] = edu_lst

    #marital encoded column
    marital_lst = []
    marital_groups = input_df.groupby('Marital_Status').mean()['Churn']

    for val in input_df['Marital_Status']:
        marital_lst.append(marital_groups.loc[val])

    input_df['Marital_Status_Churn'] = marital_lst

    #income encoded column
    income_lst = []
    income_groups = input_df.groupby('Income_Category').mean()['Churn']

    for val in input_df['Income_Category']:
        income_lst.append(income_groups.loc[val])

    input_df['Income_Category_Churn'] = income_lst

    #card encoded column
    card_lst = []
    card_groups = input_df.groupby('Card_Category').mean()['Churn']
    
    for val in input_df['Card_Category']:
        card_lst.append(card_groups.loc[val])

    input_df['Card_Category_Churn'] = card_lst

    return input_df


def perform_feature_engineering(input_df, response):
    '''
    input:
              input_df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
    
    input_df = encoder_helper(input_df, constants.CAT_COLUMNS, None)

    #Empty dataframe for features
    X = pd.DataFrame()
    y = input_df['Churn']

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = input_df[keep_cols]

    X.head()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test




def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    #again, but saving it as a figure.
    #TODO save the classification report in a local variable to avoid double computation
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=-1.8)
    lrc_plot.plot(ax=ax, alpha=-1.8)
    plt.show()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90);

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    #Random forest classifier training
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    #Logistic regression classifier training
    lrc.fit(X_train, y_train)

    #Save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    

def test_models(X_train, X_test, model_pth_rf, model_pth_lr):
    '''
    test models after completed training

    Input:
        X_train: X training data
        X_test: X testing data
        model_rf: random forest model
        model_lr: logistic regression model
    
    Output:
        TBD
    '''
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    pass