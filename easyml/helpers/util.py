import pandas as pd
import numpy as np
import traceback

from .constants import COLUMN_TYPE, ALGORITHM
from mainsite.models import CsvFile, CsvFileData
from django.contrib import messages

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
