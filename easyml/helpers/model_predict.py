import json
import pandas as pd
import numpy as np
import traceback
import msgpack
import math
import pickle
import base64

from pprint import pprint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from django.utils.encoding import smart_bytes
from .constants import COLUMN_TYPE, ALGORITHM, algorithm_name_map
from mainsite.models import CsvFile, CsvFileData, MLModel
from .util import get_dataframe


def run_model_predict(file_obj, model_obj):

    file_data = CsvFileData.objects.filter(parent_file=file_obj)\
        .exclude(type=COLUMN_TYPE.IGNORE)

    if file_data.count() == 0:
        print("Error: No data for file {}".format(file_obj.display_name))
        return

    input_data = file_data.filter(type=COLUMN_TYPE.INPUT)

    algorithm_type_num = model_obj.type_num

    model = model_obj.data
    input_df = get_dataframe(input_data)

    if algorithm_type_num == ALGORITHM.LINEAR_REGRESSION:
        run_linear_regression_model(input_df, model)

    elif algorithm_type_num == ALGORITHM.K_NEAREST_NEIGHBORS:
        run_k_nearest_neighbors_model(input_df, model)

    elif algorithm_type_num == ALGORITHM.LOGISTIC_REGRESSION:
        run_logistic_regression_model(input_df, model)

    elif algorithm_type_num == ALGORITHM.NEAREST_CENTROID:
        run_nearest_centroid(input_df, model)

    elif algorithm_type_num == ALGORITHM.LINEAR_DISCRIMINANT_ANALYSIS:
        run_linear_discriminant_analysis(input_df, model)

    elif algorithm_type_num == ALGORITHM.DECISION_TREE:
        run_decision_tree(input_df, model)

    elif algorithm_type_num == ALGORITHM.GAUSSIAN_NAIVE_BAYES:
        run_gaussian_naive_bayes(input_df, model)

    elif algorithm_type_num == ALGORITHM.RANDOM_FOREST_CLASSIFIER:
        run_random_forest_classifier(input_df, model)

    elif algorithm_type_num == ALGORITHM.RANDOM_FOREST_REGRESSOR:
        run_random_forest_regressor(input_df, model)

    elif algorithm_type_num == ALGORITHM.SUPPORT_VECTOR_MACHINE:
        run_support_vector_machine(input_df, model)


def run_linear_regression_model(input_df, model):
    print(model.coef_)
    print(model.intercept_)


def run_k_nearest_neighbors_model(input_df, model):
    pass


def run_logistic_regression_model(input_df, model):
    pass


def run_nearest_centroid(input_df, model):
    pass


def run_linear_discriminant_analysis(input_df, model):
    pass


def run_decision_tree(input_df, model):
    pass


def run_gaussian_naive_bayes(input_df, model):
    pass


def run_random_forest_classifier(input_df, model):
    pass


def run_random_forest_regressor(input_df, model):
    pass


def run_support_vector_machine(input_df, model):
    pass
