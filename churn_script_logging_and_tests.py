'''
Testing module for the churn library

Author: Patrick
Date: Oct 2022

TODO put the paths in constants.py

'''
import os
import time
import logging
import joblib
import pytest
import churn_library as cl
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')

# From the production ready code lesson, the following two functions were
# already coded by Udacity.


@pytest.fixture(scope="module", name="path")
def fixture_path():
    '''path fixture'''
    return "./data/bank_data.csv"


@pytest.fixture(scope="module", name="perform_eda")
def fixture_perform_eda():
    '''eda fixture'''
    return cl.perform_eda


@pytest.fixture(scope="module", name="encoder_helper")
def fixture_encoder_helper():
    '''encoder_helper fixture'''
    return cl.encoder_helper


@pytest.fixture(scope="module", name="perform_feature_engineering")
def fixture_perform_feature_engineering():
    '''perform_feature_engineering fixture'''
    return cl.perform_feature_engineering


@pytest.fixture(scope="module", name="classification_report_image")
def fixture_classification_report_image():
    '''classification_report_image fixture'''
    return cl.classification_report_image


@pytest.fixture(scope="module", name="train_models")
def fixture_train_models():
    '''train_models fixture'''
    return cl.train_models


@pytest.fixture(scope="module", name="fn_test_models")
def fixture_fn_test_models():
    '''test_models fixture'''
    return cl.test_models


@pytest.fixture(scope="module", name="feature_importance_plot")
def fixture_feature_importance_plot():
    '''feature_importance_plot fixture'''
    return cl.feature_importance_plot


def check_file_integrity(path, start_time, end_time):
    '''
    Checks if the file provied by path exists and was modified between two timepoints

    Input:
            path: str, filepath of the file to be checked
            start_time: float, start time of the function call that creates file
            end_time: float, end time of the function call that creates file

    Raises:
            AssertionError: is raised when mtime is not between start_time and end_time
            FileNotFoundError: is raised when path does not point to a file
    '''
    file_handle = open(path, 'r', encoding='utf-8')
    file_handle.close()

    file_mtime = os.path.getmtime(path)
    assert file_mtime >= start_time and file_mtime <= end_time


def time_unit(func, args=None, kwargs=None):
    '''
    Times an a unit run

    Input:
            func: a callable function
            args: (optional), arguments to be passed to func
            kwargs: (optional), keyword arguments to be passed to func

    Returns:
            ret_val: The return value of the function
            start_time: float, the starting time of the unit run
            end_time: float, the end time of the unit run
    '''

    start_time = time.time()

    # Need to do that, otherwise the run will fail due to incorrect types
    if args is None and kwargs is None:
        ret_val = func()
    elif args is not None and kwargs is None:
        ret_val = func(*args)
    elif args is None and kwargs is not None:
        ret_val = func(**kwargs)
    elif args is not None and kwargs is not None:
        ret_val = func(*args, **kwargs)

    end_time = time.time()

    return ret_val, start_time, end_time


def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        input_df = cl.import_data(path)
        pytest.input_df = input_df
        #request.config.cache.set('input_df', input_df)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert input_df.shape[0] > 0
        assert input_df.shape[1] > 0
        logging.info("Testing import_data: Data consistent")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    start_time = None
    end_time = None
    # Load the dataframe from cache (will be better once dataframes get bigger)
    try:
        input_df = pytest.input_df
        _, start_time, end_time = time_unit(perform_eda, (input_df,))
        logging.info("Performing EDA: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: Plots could not be saved, check that the ./images/eda/ path exists.")
        raise err

    try:
        check_file_integrity(constants.GENDER_PLT_PTH, start_time, end_time)
        check_file_integrity(constants.AGE_PLT_PTH, start_time, end_time)
        check_file_integrity(constants.CORR_PLT_PTH, start_time, end_time)

        logging.info('Testing perform_eda: SUCCESS')

    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: No plots could be found. Check filepaths.")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The current run did not create at least one of the reports.")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    # Check that the *_Churn columns exists and their value is intact, e.g.
    # not null
    try:
        input_df = pytest.input_df
        result_df = encoder_helper(input_df, constants.CAT_COLUMNS, None)

        # Access all of the categorical columns
        cat_columns_with_churn = [x + "_Churn" for x in constants.CAT_COLUMNS]
        result_df[cat_columns_with_churn]

        # Check that the data is intact
        assert not result_df[cat_columns_with_churn].isnull().any().any()

        # Save result for the perform feature engineering test
        pytest.encoder_helper_result_df = result_df

        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing encoder_helper: At least one of the proprtional churn columns is missing.")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Data is corrupted, at least one column has a null entry.")
        logging.error(
            "\n%s", str(result_df[cat_columns_with_churn].isnull().any()))
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        input_df = pytest.input_df
        (features_train, features_test, 
            targets_train, targets_test) = perform_feature_engineering(
            input_df, None)

        # Check that the split train/test data is intact
        assert features_train.shape[0] > 0
        assert features_test.shape[0] > 0
        assert targets_train.shape[0] > 0
        assert targets_test.shape[0] > 0

        # Features should have the same number of rows and the train split
        # should have more rows than the test split
        # Same amount of features
        assert features_train.shape[0] > features_test.shape[0]
        assert features_train.shape[1] == features_test.shape[1]

        # Same number of samples between features and target
        assert features_train.shape[0] == targets_train.shape[0]
        assert features_test.shape[0] == targets_test.shape[0]

        pytest.X_train = features_train
        pytest.X_test = features_test
        pytest.y_train = targets_train
        pytest.y_test = targets_test

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering: There is a missing column.")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The train/test split failed and the data is corrupted")
        raise err


def test_test_models(fn_test_models):
    '''
    test test_models
    '''

    try:
        # Run the unit
        features_train = pytest.X_train  # request.config.cache.get('X_train')
        features_test = pytest.X_test  # request.config.cache.get('X_test')

        (targets_train_preds_rf, targets_test_preds_rf, targets_train_preds_lr,
         targets_test_preds_lr) = fn_test_models(features_train, features_test,
                             constants.RFC_MODEL_PTH, constants.LR_MODEL_PTH)

        # Check data integrity
        assert targets_train_preds_lr.shape[0] == features_train.shape[0]
        assert targets_train_preds_rf.shape[0] == features_train.shape[0]
        assert targets_test_preds_lr.shape[0] == features_test.shape[0]
        assert targets_test_preds_rf.shape[0] == features_test.shape[0]

        #Store in namespace
        pytest.y_train_preds_rf = targets_train_preds_rf
        pytest.y_train_preds_lr = targets_train_preds_lr
        pytest.y_test_preds_rf = targets_test_preds_rf
        pytest.y_test_preds_lr = targets_test_preds_lr

        logging.info("Testing test_models: SUCCESS")

    except AssertionError as err:
        logging.error('Testing test_models: Shape mismatch, data is corrupted')
        raise err


def test_classification_report_image(classification_report_image):
    '''
    test classification_report_image

    Checks if the paths of the desired report images exist
    '''
    try:

        # Get the necessary arguments
        targets_train = pytest.y_train
        # request.config.cache.get('y_train')
        targets_train_preds_lr = pytest.y_train_preds_lr
        # request.config.cache.get('y_train_preds_lr')
        targets_train_preds_rf = pytest.y_train_preds_rf
        # request.config.cache.get('y_train_preds_rf')

        targets_test = pytest.y_test  # request.config.cache.get('y_test')
        # request.config.cache.get('y_test_preds_lr')
        targets_test_preds_lr = pytest.y_test_preds_lr
        # request.config.cache.get('y_test_preds_rf')
        targets_test_preds_rf = pytest.y_test_preds_rf

        # Run the unit
        _, start_time, end_time = time_unit(classification_report_image, (
            targets_train, targets_test, targets_train_preds_lr, targets_train_preds_rf, targets_test_preds_lr, targets_test_preds_rf))

        # Check the file integrity
        check_file_integrity(
            constants.CLS_REPORT_PLT_PTH,
            start_time,
            end_time)
        check_file_integrity(constants.ROC_CURVE_PLT_PTH, start_time, end_time)

        logging.info("Testing classification_report_image: SUCCESS")

    except FileNotFoundError as err:
        logging.error(
            "Testing classification_report_image: No plots could be found. Check filepaths.")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: The current run did not create at least one of the reports.")
        raise err


def test_feature_importance_plot(feature_importance_plot):
    '''
    Test feature_importance_plot by checking if paths exists and stuff has been created
    '''
    try:
        # Run the unit

        # Load the rf model from disk

        rfc = joblib.load(constants.RFC_MODEL_PTH)
        features_test = pytest.X_test
        _, start_time, end_time = time_unit(feature_importance_plot, (
            rfc, features_test, constants.FEATURE_IMPORTANCE_PLT_RFC_PTH, constants.FEATURE_IMPORTANCE_SHAP_PLT_RFC_PTH))

        # Do the file integrity test
        check_file_integrity(constants.FEATURE_IMPORTANCE_PLT_RFC_PTH,
                             start_time, end_time)
        check_file_integrity(constants.FEATURE_IMPORTANCE_SHAP_PLT_RFC_PTH,
                             start_time, end_time)

        logging.info("Testing feature_importance_plot: SUCCESS")

    except FileNotFoundError as err:
        logging.error(
            "Testing feature_importance_plot: No plots could be found. Check filepaths.")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: The current run did not create at least one of the reports.")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        # Time the unit
        features_train = pytest.X_train
        features_test = pytest.X_test
        targets_train = pytest.y_train
        targets_test = pytest.y_test

        _, start_time, end_time = time_unit(
            train_models, (features_train, features_test, 
                        targets_train, targets_test))

        check_file_integrity(
            './models/logistic_model.pkl',
            start_time,
            end_time)
        check_file_integrity('./models/rfc_model.pkl', start_time, end_time)

        logging.info("Testing train_models: SUCCESS")

    except FileNotFoundError as err:
        logging.error(
            "Testing train_models: No models could be found. Check filepaths.")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing train_models: The current run did not create at least one of the models.")
        raise err


if __name__ == "__main__":

    # Run the pipeline
    bank_data_df = cl.import_data(constants.DATA_PATH)

    # Alters the dataframe in place
    cl.perform_eda(bank_data_df)

    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        bank_data_df)

    cl.train_models(X_train, X_test, y_train, y_test)
