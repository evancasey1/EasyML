import pandas as pd
import numpy as np
import traceback

from .constants import COLUMN_TYPE, ALGORITHM
from mainsite.models import *
from helpers.constants import *
from django.contrib import messages
from operator import itemgetter
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext as _

class Echo:
    def write(self, value):
        """Write the value by returning it, instead of storing in a buffer."""
        return value

def validate_password_strength(value1, value2):
    """Validates that a password is as least 7 characters long and has at least
    1 digit and 1 letter.
    """
    min_length = 7

    if value1 != value2:
        return False, 'Passwords must match.'

    if len(value1) < min_length:
        return False, 'Password must be at least {0} characters long.'.format(min_length)

    # check for digit
    if not any(char.isdigit() for char in value1):
        return False, 'Password must contain at least 1 digit.'

    # check for letter
    if not any(char.isalpha() for char in value1):
        return False, 'Password must contain at least 1 letter.'

    return True, ""


def get_dataframe(qry_data):
    column_headers = qry_data.values_list('column_header', flat=True).distinct()
    column_data = {}
    for header in column_headers:
        header_data = qry_data.filter(column_header=header).order_by('row_num')
        column_data[header] = [d.data for d in header_data]

    return pd.DataFrame.from_dict(column_data)


# Gets the map of data for a file to their placeholders
# Int to String
def get_itos_map(file_id):
    file_obj = CsvFile.objects.get(id=file_id)
    placeholder_data = CsvFileData.objects.filter(parent_file=file_obj).exclude(placeholder=None)

    itos_map = {}
    for pdata in placeholder_data:
        header = pdata.column_header
        if header not in itos_map:
            itos_map[header] = {}

        placeholder = pdata.placeholder
        if placeholder not in itos_map[header]:
            itos_map[header][pdata.data] = placeholder

    return itos_map

# Gets the map of placeholders for a file to their data
# String to Int
def get_stoi_map(file_id):
    file_obj = CsvFile.objects.get(id=file_id)
    placeholder_data = CsvFileData.objects.filter(parent_file=file_obj).exclude(placeholder=None)

    stoi_map = {}
    for pdata in placeholder_data:
        header = pdata.column_header
        if header not in stoi_map:
            stoi_map[header] = {}

        placeholder = pdata.placeholder
        if placeholder not in stoi_map[header]:
            stoi_map[header][placeholder] = pdata.data

    return stoi_map


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
