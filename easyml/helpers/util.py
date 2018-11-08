import pandas as pd
import numpy as np
import traceback

from .constants import COLUMN_TYPE, ALGORITHM
from mainsite.models import *
from helpers.constants import *
from django.contrib import messages
from operator import itemgetter

class Echo:
    def write(self, value):
        """Write the value by returning it, instead of storing in a buffer."""
        return value

def get_dataframe(qry_data):
    column_headers = qry_data.values_list('column_header', flat=True).distinct()
    column_data = {}
    for header in column_headers:
        header_data = qry_data.filter(column_header=header).order_by('row_num')
        column_data[header] = [d.data for d in header_data]

    return pd.DataFrame.from_dict(column_data)


def set_column_types(file_id, header_map):
    des_values = list(header_map.values())

    if COLUMN_TYPE.TARGET not in des_values:
        raise Exception("A target column is required")

    if COLUMN_TYPE.INPUT not in des_values:
        raise Exception("An input column is required")

    if des_values.count(COLUMN_TYPE.TARGET) > 1:
        raise Exception("Only one target column is allowed")

    csv_data = CsvFileData.objects.filter(parent_file_id=file_id)
    for header in header_map:
        csv_data.filter(column_header=header).update(type=header_map[header])


def get_user_files(user):
    return CsvFile.objects.filter(file_owner=user)


def get_user_models(user):
    user_files = get_user_files(user)
    return MLModel.objects.filter(parent_file__in=user_files)

def get_alg_lst():
    alg_lst = []
    for alg in ALGORITHM_NAME_MAP:
        alg_lst.append({
            'num': int(alg),
            'name': ALGORITHM_NAME_MAP[alg]
        })

    return sorted(alg_lst, key=itemgetter('num'))

def get_r2(y_pred, y_test):
    n = len(y_test)
    y_bar = sum(y_test) / n
    ss_tot = 0
    ss_res = 0
    for i in range(n):
        ss_tot += (y_test[i] - y_bar)**2
        ss_res += (y_test[i] - y_pred[i])**2

    return round(1 - (ss_res/ss_tot), 4)

def get_match_acc(y_pred, y_test):
    n = len(y_test)
    if n == 0:
        return 0

    correct = 0
    for i in range(n):
        if y_pred[i] == y_test[i]:
            correct += 1

    return round((correct / n) * 100.0, 4)

def to_percent(val, n=4):
    return round((val * 100), n)
