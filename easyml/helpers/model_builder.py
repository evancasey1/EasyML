import json
import pandas as pd
import numpy as np
import traceback
import msgpack
import pickle

from pprint import pprint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .constants import COLUMN_TYPE, ALGORITHM
from mainsite.models import CsvFile, CsvFileData, MLModel
from .util import get_dataframe


def create_model(algorithm_type, file_id):
    file_data = CsvFileData.objects.filter(parent_file_id=file_id)\
        .exclude(type=COLUMN_TYPE.IGNORE)

    if file_data.count() == 0:
        print("Error: No data for file {}".format(file_id))
        return

    input_data = file_data.filter(type=COLUMN_TYPE.INPUT)
    target_data = file_data.filter(type=COLUMN_TYPE.TARGET)

    model = None
    name = None

    input_df = get_dataframe(input_data)
    target_df = get_dataframe(target_data)

    if algorithm_type == ALGORITHM.LINEAR_REGRESSION:
        name = "Linear Regression"
        model = create_linear_regression_model(input_df, target_df)

    elif algorithm_type == ALGORITHM.K_NEAREST_NEIGHBORS:
        name = "K Nearest Neighbors"
        model = create_k_nearest_neightbors_model(input_df, target_df)

    elif algorithm_type == ALGORITHM.LOGISTIC_REGRESSION:
        name = "Logistic Regression"
        model = create_logistic_regression_model(input_df, target_df)

    elif algorithm_type == ALGORITHM.NEAREST_CENTROID:
        name = "Nearest Centroid"
        model = create_nearest_centroid(input_df, target_df)

    if model:
        save_model(model, name, file_id)

def save_model(model, name, file_id):
    serialized_data = pickle.dumps(model)

    model_obj = MLModel()
    model_obj.name = name
    model_obj.data = serialized_data
    model_obj.parent_file = CsvFile.objects.get(id=file_id)
    model_obj.save()

def create_linear_regression_model(input_df, target_df):
    lin_reg = LinearRegression().fit(input_df, target_df)
    print("Coef:", lin_reg.coef_)
    print("Intercept:", lin_reg.intercept_)

    return lin_reg

def create_logistic_regression_model(input_df, target_df):
    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)
    steps = [('std_scaler', StandardScaler())]
    steps += [('log_regression', LogisticRegression(penalty='l2'))]
    pipe = Pipeline(steps)

    parameters = {'log_regression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5)

    clf = gs.fit(x_train, y_train)
    pprint("Best params:", clf.best_params_)

    clf = gs.fit(input_df, target_df)
    return clf

def create_k_nearest_neightbors_model(input_df, target_df):
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(input_df)
    distances, indices = neighbors.kneighbors(input_df)
    print("Distances:", distances)
    print("Indices:", indices)

    return neighbors

def create_nearest_centroid(input_df, target_df):
    x_train, x_valid, y_train, y_valid = train_test_split(input_df, target_df, test_size=0.20)
    clf = NearestCentroid()

    clf.fit(x_train, y_train)
    print("Predict: ", clf.predict(x_valid))
    print()
    print("Score: ", clf.score(x_train, y_train))

    # Final model
    clf.fit(input_df, target_df)

    return clf

