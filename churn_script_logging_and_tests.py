'''
Testing module for the churn library

Author: Patrick
Date: Oct 2022

TODO put the paths in constants.py

'''
import os
import time
import constants
import logging
import pytest
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#From the production ready code lesson, the following two functions were already coded by Udacity.
@pytest.fixture(scope="module")
def path():
    return "./data/bank_data.csv"

@pytest.fixture(scope="module")
def perform_eda():
	return cl.perform_eda

@pytest.fixture(scope="module")
def encoder_helper():
	return cl.encoder_helper

@pytest.fixture(scope="module")
def perform_feature_engineeering():
	return cl.perform_feature_engineering

@pytest.fixture(scope="module")
def classification_report_image():
	return cl.classification_report_image

@pytest.fixture(scope="module")
def train_models():
	return cl.train_models


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
	file_handle = open(path, 'r')
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
	
	#Need to do that, otherwise the run will fail due to incorrect types
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

def test_import(path, request):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		input_df = cl.import_data(path)
		request.config.cache.set('input_df', input_df)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert input_df.shape[0] > 0
		assert input_df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, request):
	'''
	test perform eda function
	'''
	start_time = None
	end_time = None
	#Load the dataframe from cache (will be better once dataframes get bigger)
	try:
		input_df = request.config.cache.get('input_df')	
		_, start_time, end_time = time_unit(perform_eda, (input_df))
		logging.info("Performing EDA: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing perform_eda: Plots could not be saved, check that the ./images/eda/ path exists.")
		raise err
	
	try:
		check_file_integrity("./images/eda/gender.png", start_time, end_time)
		check_file_integrity("./images/eda/age.png", start_time, end_time)
		check_file_integrity("./images/eda/correlation.png", start_time, end_time)

	except FileNotFoundError as err:
		logging.error("Testing perform_eda: No plots could be found. Check filepaths.")	
		raise err
	except AssertionError as err:
		logging.error("Testing perform_eda: The current run did not create at least one of the reports.")
		raise err

def test_encoder_helper(encoder_helper, request):
	'''
	test encoder helper
	'''
	#Check that the *_Churn columns exists and their value is intact, e.g. not null
	try:
		input_df = request.config.cache.get("input_df")		
		result_df = encoder_helper(input_df, constants.CAT_COLUMNS, None)

		#Access all of the categorical columns
		cat_columns_with_churn = [x + "_Churn" for x in constants.CAT_COLUMNS]
		result_df[cat_columns_with_churn]

		#Check that the data is intact
		assert result_df[cat_columns_with_churn].isnull().any().any()

		#Save result for the perform feature engineering test
		request.config.cache.set("encoder_helper_result_df", result_df)

	except KeyError as err:
		logging.error("Testing encoder_helper: At least one of the proprtional churn columns is missing.")
		raise err
	except AssertionError as err:
		logging.error("Testing encoder_helper: Data is corrupted, at least one column has a null entry.")
		raise err



def test_perform_feature_engineering(perform_feature_engineering, request):
	'''
	test perform_feature_engineering
	'''
	try:
		input_df = request.config.cache.get("input_df")
		X_train, X_test, y_train, y_test = perform_feature_engineering(input_df, None)
		
		#Check that the split train/test data is intact
		assert X_train.shape[0] > 0
		assert X_test.shape[0] > 0
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0
		
		#Features should have the same number of rows and the train split
		#should have more rows than the test split
		#Same amount of features 
		assert X_train.shape[0] > X_test.shape[0]
		assert X_train.shape[1] == X_test.shape[1]

		#Same number of samples between features and target
		assert X_train.shape[0] == y_train.shape[0]
		assert X_test.shape[0] == y_test.shape[0]
	
	except KeyError as err:
		logging.error("Testing feature engineering: There is a missing column.")
		raise err

	except AssertionError as err:
		logging.error("Testing feature engineering: The train/test split failed and the data is corrupted")
		raise err

def test_classification_report_image(classification_report_image, request):
	'''
	test classification_report_image

	Checks if the paths of the desired report images exist
	'''
	try:
		
		#Run the unit
		#TODO figure out where to get these arguments from
		_, start_time, end_time = time_unit(classification_report_image, (y_train, y_test,
									y_train_preds_lr, y_train_preds_rf,
									y_test_preds_lr, y_test_preds_rf))

		#Check the file integrity
		check_file_integrity('./images/results/random_forest_report.png', start_time, end_time)
		check_file_integrity('./images/results/logistic_regression_report.png', start_time, end_time)
		check_file_integrity('./images/results/roc_curve.png', start_time, end_time)
		
	except FileNotFoundError as err:
		logging.error("Testing classification_report_image: No plots could be found. Check filepaths.")	
		raise err
	
	except AssertionError as err:
		logging.error("Testing classification_report_image: The current run did not create at least one of the reports.")
		raise err

def test_feature_importance_plot(feature_importance_plot):
	'''
	Test feature_importance_plot by checking if paths exists and stuff has been created
	'''
	try:
		#Run the unit
		_, start_time, end_time = time_unit(feature_importance_plot, (model, X_data, output_pth))

		#Do the file integrity test
		check_file_integrity('./images/results/feature_importances.png', start_time, end_time)
		check_file_integrity('./images/results/feature_importances_shap.png', start_time, end_time)

	except FileNotFoundError as err:
		logging.error("Testing feature_importance_plot: No plots could be found. Check filepaths.")	
		raise err
	
	except AssertionError as err:
		logging.error("Testing feature_importance_plot: The current run did not create at least one of the reports.")
		raise err

def test_train_models(train_models):
	'''
	test train_models
	'''
	try:
		#Time the unit
		_, start_time, end_time = time_unit(train_models, (X_train, X_test, y_train, y_test))

		check_file_integrity('./models/logistic_model.pkl', start_time, end_time)
		check_file_integrity('./models/rfc_model.pkl', start_time, end_time)

	except FileNotFoundError as err:
		logging.error("Testing train_models: No models could be found. Check filepaths.")	
		raise err
	
	except AssertionError as err:
		logging.error("Testing train_models: The current run did not create at least one of the models.")
		raise err

def test_test_models(test_models):
	'''
	test test_models
	TODO: to be implemented later once figured out what is to be tested
	'''
	pass


if __name__ == "__main__":
	pass








