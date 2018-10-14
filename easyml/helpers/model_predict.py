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
        .exclude(type=COLUMN_TYPE.IGNORE).order_by('column_num')

    print(file_data.count())

    if file_data.count() == 0:
        print("Error: No data for file {}".format(file_obj.display_name))
        return

    input_data = file_data.filter(type=COLUMN_TYPE.INPUT).order_by('column_num')
    target_col = file_data.filter(type=COLUMN_TYPE.TARGET).first().column_header

    model = model_obj.data
    input_df = get_dataframe(input_data)

    results = pd.DataFrame(model.predict(input_df), columns=[target_col])

    concat_df = pd.concat([input_df, results], axis=1)
    csv_data = concat_df.to_csv()

    print(csv_data)
    return csv_data
