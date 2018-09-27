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
