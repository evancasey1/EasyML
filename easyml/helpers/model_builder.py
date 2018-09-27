import pandas as pd
import numpy as np
import traceback

from .constants import COLUMN_TYPE, ALGORITHM
from mainsite.models import CsvFile, CsvFileData
from .util import get_dataframe
from django.contrib import messages


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

def create_linear_regression_model(input_df, target_df):
    print("create_linear_regression_model")
    print(input_df)
    print(target_df)

def create_k_nearest_neightbors_model(input_df, target_df):
    print("create_k_nearest_neightbors_model")
