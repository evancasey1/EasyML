import pandas as pd
import numpy as np
import traceback

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from .constants import COLUMN_TYPE, ALGORITHM
from mainsite.models import CsvFile, CsvFileData
from .util import get_dataframe


def create_model(algorithm_type, file_id):
    file_data = CsvFileData.objects.filter(parent_file_id=file_id)\
        .exclude(type=COLUMN_TYPE.IGNORE)

    if file_data.count() == 0:
        print("Error: No data for file {}".format(file_id))
        return

    input_data = file_data.filter(type=COLUMN_TYPE.INPUT)
    target_data = file_data.filter(type=COLUMN_TYPE.TARGET)

    input_df = get_dataframe(input_data)
    target_df = get_dataframe(target_data)

    if algorithm_type == ALGORITHM.LINEAR_REGRESSION:
        create_linear_regression_model(input_df, target_df)

    elif algorithm_type == ALGORITHM.K_NEAREST_NEIGHBORS:
        create_k_nearest_neightbors_model(input_df, target_df)

    elif algorithm_type == ALGORITHM.LOGISTIC_REGRESSION:
        create_logistic_regression_model(input_df, target_df)

def create_linear_regression_model(input_df, target_df):

    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)

    lin_reg = LinearRegression().fit(x_train, y_train)


def create_logistic_regression_model(input_df, target_df):
    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)
    steps = [('std_scaler', StandardScaler())]
    steps += [('log_regression', LogisticRegression(penalty='l2'))]
    pipe = Pipeline(steps)

    parameters = {'log_regression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5)

    clf = gs.fit(x_train, y_train)
    best_c = clf.best_params_['log_regression__C']

def create_k_nearest_neightbors_model(input_df, target_df):
    print("create_k_nearest_neightbors_model")
