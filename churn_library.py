# library doc string
'''
This module contains functions to perform data analysis and machine learning on
the Kaggle Credit Card customers dataset to predict customers who are likely to churn.

Author: Patrick
Date: October 2022
'''

# import libraries
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import joblib
import shap
import matplotlib
import constants


# Some setups
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()
# Use non-interactive backend for plotting
matplotlib.use('agg')


def print_and_log(message, log_fn):
    '''
    Prints to the console and logs to log

    Input:
        message: Message to be printed and logged
        log_fn: Log function object, e.g. logging.info or logging.debug
    '''
    print(message)
    log_fn(message)


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
        print_and_log(f"Could not read file {input_pth}", logging.error)


def perform_eda(input_df):
    '''
    perform eda on df and save figures to images folder
    prints out to console / log
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Get the EDA attributes
    shape = input_df.shape
    null_count_per_column = input_df.isnull().sum()
    df_summary = input_df.describe()

    # Print out the attributes
    print_and_log(
        f"Dataframe has {shape[0]} rows and {shape[1]} columns.",
        logging.info)
    print_and_log(
        "The amount of rows with null values per column are as follows:",
        logging.info)
    print_and_log("\n" + str(null_count_per_column), logging.info)
    print_and_log("The summary of the dataframe is as follows:", logging.info)
    print_and_log("\n" + str(df_summary), logging.info)

    # Save the plots (this seems kind of repetitive)
    # Univariate categorical
    plt.figure(figsize=(20, 10))
    input_df['Gender'].hist()
    plt.xlabel("Gender")
    plt.ylabel("Count")
    # Make the plot fit in the figure
    plt.tight_layout()
    # Save to disk
    plt.savefig(constants.GENDER_PLT_PTH)
    print_and_log(
        f"Created gender histogram under {constants.GENDER_PLT_PTH}",
        logging.info)

    # Univariate quantitative
    plt.figure(figsize=(20, 10))
    input_df['Customer_Age'].hist()
    plt.xlabel("Age")
    plt.ylabel("Count")
    # Make the plot fit in the figure
    plt.tight_layout()
    # Save to disk
    plt.savefig(constants.AGE_PLT_PTH)
    print_and_log(
        f"Created age histogram under {constants.AGE_PLT_PTH}",
        logging.info)

    # Bivariate
    plt.figure(figsize=(20, 10))
    sns.heatmap(input_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    # Make plot fit in the figure
    plt.tight_layout()
    # Save to disk
    plt.savefig(constants.CORR_PLT_PTH)
    print_and_log(
        f"Created correlation plot under {constants.CORR_PLT_PTH}",
        logging.info)

    print_and_log("EDA finished.", logging.info)


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
    print_and_log("Encoding columns....", logging.info)

    # Create Churn column: Only 0 if existing customer, else 1
    input_df['Churn'] = input_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Do the encoding by looping over all the categorical columns
    # and then looping over the groups created by groupby to assign the
    # Churn value
    for col in category_lst:
        group_churn = input_df.groupby(col).mean()['Churn']
        for group in group_churn.index:
            input_df.loc[input_df[col] == group,
                         col + '_Churn'] = group_churn[group]

    print_and_log("Encoded columns", logging.info)
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

    print_and_log("Performing feature engineering", logging.info)
    input_df = encoder_helper(input_df, constants.CAT_COLUMNS, None)

    # Empty dataframe for features
    X = pd.DataFrame()
    y = input_df['Churn']

    
    X[constants.KEEP_COLS] = input_df[constants.KEEP_COLS]

    X.head()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print_and_log("Performed feature engineering", logging.info)

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
    print_and_log('random forest results', logging.info)
    print_and_log('test results', logging.info)
    print_and_log(
        "\n" + str(classification_report(y_test, y_test_preds_rf)), logging.info)
    print_and_log('train results', logging.info)
    print_and_log(
        "\n" + str(classification_report(y_train, y_train_preds_rf)), logging.info)

    print_and_log('logistic regression results', logging.info)
    print_and_log('test results', logging.info)
    print_and_log(
        "\n" + str(classification_report(y_test, y_test_preds_lr)), logging.info)
    print_and_log('train results', logging.info)
    print_and_log(
        "\n" + str(classification_report(y_train, y_train_preds_lr)), logging.info)

    # again, but saving it as a figure.
    # TODO save the classification report in a local variable to avoid double
    # computation
    print_and_log("Creating graphical classification report", logging.info)
    plt.figure(figsize=(7, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(constants.CLS_REPORT_PLT_PTH)
    print_and_log(
        f"Saved graphical report to {constants.CLS_REPORT_PLT_PTH}",
        logging.info)

    print_and_log("Creating ROC curves", logging.info)
    lrc_plot = RocCurveDisplay.from_predictions(y_test, y_test_preds_lr)
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_predictions(y_test, y_test_preds_rf,
                                                ax=ax, alpha=0.8)
    plt.tight_layout()
    plt.savefig(constants.ROC_CURVE_PLT_PTH)
    print_and_log(
        f"Saved ROC curve plot to {constants.ROC_CURVE_PLT_PTH}",
        logging.info)


def feature_importance_plot(model, X_data, output_pth, shap_output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: Path for the feature importance plot to save
            shap_output_pth: Path for the shap feature importance plot to save

    output:
             None
    '''
    print_and_log("Running SHAP analysis...", logging.info)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    plt.figure(figsize=(15, 10))
    shap.summary_plot(shap_values, X_data, plot_type="bar")

    # Make labels fit
    plt.tight_layout()
    plt.savefig(shap_output_pth)
    print_and_log(f"Saved shap plot under {shap_output_pth}", logging.info)

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Make the labels fit
    plt.tight_layout()

    plt.savefig(output_pth)
    print_and_log(
        f"Saved feature importance plot under {output_pth}",
        logging.info)


def test_models(X_train, X_test, model_pth_rf, model_pth_lr):
    '''
    test models after completed training

    Input:
        X_train: X training data
        X_test: X testing data
        model_rf: random forest model
        model_lr: logistic regression model

    Output:
        y_train_preds_rf: Prediction result for the random forest classifier
                            on the train split
        y_test_preds_rf: Prediction result for the random forest classifier
                            on the test split
        y_train_preds_lr: Prediction result for the log reg classifier
                            on the train split
        y_test_preds_lr: Prediction result for the log reg classifier
                            on the test split

    '''

    # Load the models from disk
    print_and_log(f"Loading model from path {model_pth_rf}", logging.info)
    rfc = joblib.load(model_pth_rf)

    print_and_log(f"Loading model from path {model_pth_lr}", logging.info)
    lrc = joblib.load(model_pth_lr)

    print_and_log(
        "Predicting values with random forest classifier",
        logging.info)
    # Do prediction for random forest
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    print_and_log(
        "Predicting values from logistic regression classifier",
        logging.info)
    # Do prediction for logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


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
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Random forest classifier training
    print_and_log(
        "Performing Random Forest Grid search with n_jobs=12",
        logging.info)
    print_and_log(
        "Decrease this value if system becomes irresponsive.",
        logging.info)
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        n_jobs=12)
    cv_rfc.fit(X_train, y_train)
    print_and_log("Finished grid search of rf classifier", logging.info)

    # Logistic regression classifier training
    print_and_log("Training LR classifier", logging.info)
    lrc.fit(X_train, y_train)
    print_and_log("Trained LR classifier", logging.info)

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, constants.RFC_MODEL_PTH)
    print_and_log(
        f"Dumped RFC model under {constants.RFC_MODEL_PTH}",
        logging.info)
    joblib.dump(lrc, constants.LR_MODEL_PTH)
    print_and_log(
        f"Dumped LR model under {constants.RFC_MODEL_PTH}",
        logging.info)

    # Use the model. Formula is WX + b = y, used by the predict function
    (y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr) = test_models(
        X_train, X_test, constants.RFC_MODEL_PTH, constants.LR_MODEL_PTH)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Only do it with the random forest classifier because it SHAP does not
    # support Logistic Regression classifiers
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_test,
        constants.FEATURE_IMPORTANCE_PLT_RFC_PTH,
        constants.FEATURE_IMPORTANCE_SHAP_PLT_RFC_PTH)
