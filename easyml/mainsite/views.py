import csv
import base64
import pandas as pd
import numpy as np
import traceback
import io
import matplotlib
import matplotlib.pyplot as plt
import PIL, PIL.Image

from matplotlib import pylab
from pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg

from helpers.constants import COLUMN_TYPE, ALGORITHM_NAME_MAP, PLOT_FEATURE_CAP
from helpers.model_builder import create_model
from helpers.model_predict import run_model_predict
from helpers.util import *

from .models import CsvFile, CsvFileData, MLModel
from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.http import StreamingHttpResponse
from django.urls import reverse_lazy, reverse
from django.views import generic
from .forms import CustomUserCreationForm

# Create your views here.
def index(request):
    return HttpResponseRedirect("/easyml")

class SignUp(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'

def upload_csv(request, next=None):
    data = {}
    if "GET" == request.method:
        return render(request, "upload_csv.html", data)

    csv_obj_id = None
    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        file_type = request.POST.get('file_type')

        # if file is too large, return
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)" % (csv_file.size / (1000 * 1000),))
            return HttpResponseRedirect(reverse("upload_csv"))

        ext_index = csv_file.name.find('.')
        if ext_index == -1:
            csv_name = csv_file.name
        else:
            csv_name = csv_file.name[0:ext_index]

        user = request.user
        if CsvFile.objects.filter(raw_name=csv_file.name, file_owner=user).exists():
            csv_name = (csv_name + ' (%s)') % CsvFile.objects.filter(raw_name=csv_file.name, file_owner=user).count()

        csv_obj = CsvFile(raw_name=csv_file.name, display_name=csv_name, file_owner=user)
        csv_obj.save()
        csv_obj_id = csv_obj.id

        if file_type == 'comma':
            file_data = pd.read_csv(csv_file)
        elif file_type == 'tab':
            file_data = pd.read_csv(csv_file, sep='\t')
        else:
            file_data = pd.read_excel(csv_file)

        columns = list(file_data.columns.values)

        csv_stoi_map = {}
        csv_data = []
        for row_index, row in file_data.iterrows():
            for i in range(len(columns)):
                header = columns[i]
                if 'unnamed' in header.lower():
                    continue
                data_obj = CsvFileData(parent_file=csv_obj)

                if not isinstance(row[header], int) and not isinstance(row[header], float):
                    str_val = row[header]
                    data_obj.placeholder = str_val

                    if header not in csv_stoi_map:
                        csv_stoi_map[header] = {}
                    if str_val not in csv_stoi_map[header]:
                        row_data = len(csv_stoi_map[header].keys())
                        csv_stoi_map[header][str_val] = row_data

                    data_obj.data = csv_stoi_map[header][str_val]

                else:
                    data_obj.data = row[header]

                data_obj.row_num = row_index
                data_obj.column_num = i
                data_obj.column_header = header
                csv_data.append(data_obj)

            if len(csv_data) > 500:
                CsvFileData.objects.bulk_create(csv_data)
                csv_data = []

        CsvFileData.objects.bulk_create(csv_data)

    except Exception as e:
        if csv_obj_id and CsvFile.objects.filter(id=csv_obj_id).count() > 0:
            CsvFile.objects.get(id=csv_obj_id).delete()
        messages.error(request, "Unable to upload file. " + repr(e))
        return HttpResponseRedirect('/easyml/upload/csv')

    #messages.success(request, "File successfully uploaded")

    return HttpResponseRedirect('/easyml/')

def manage_data(request):
    context = {}
    valid_files = get_user_files(request.user)

    context['valid_files'] = valid_files

    return render(request, 'manage_data.html', context=context)

def manage_models(request):
    context = {}
    valid_models = get_user_models(request.user)

    context['valid_models'] = valid_models

    return render(request, 'manage_models.html', context=context)

def delete_file(request, file_id=None):
    if not file_id:
        messages.error(request, "Unable to delete file - Invalid File ID")
        return HttpResponseRedirect('manage_data.html')

    file_id = int(file_id)
    file_obj = CsvFile.objects.get(id=file_id)

    if not file_obj.file_owner == request.user:
        messages.error(request, "Unable to delete file - Invalid Permissions")
        return HttpResponseRedirect('manage_data.html')

    file_obj.delete()
    messages.success(request, "File deleted successfully")
    return HttpResponseRedirect('/easyml/manage/data')


def rename_file(request):
    if "GET" == request.method:
        return render(request, "manage_data.html", {})

    file_id = request.POST.get('file_id')
    new_name = request.POST.get('display_name')

    if not (request.user == CsvFile.objects.get(id=file_id).file_owner):
        return HttpResponseRedirect('/easyml/manage/data')

    if CsvFile.objects.filter(file_owner=request.user, display_name=new_name).count() > 0:
        messages.error(request, "A file with that name already exists")
        return HttpResponseRedirect('/easyml/manage/data')

    file_obj = CsvFile.objects.get(id=file_id)
    file_obj.display_name = new_name
    file_obj.save()

    messages.success(request, "File successfully renamed")
    return HttpResponseRedirect('/easyml/manage/data')

def delete_model(request, model_id=None):
    if not model_id:
        messages.error(request, "Unable to delete model - Invalid Model ID")
        return HttpResponseRedirect('manage_models.html')

    model_id = int(model_id)
    model_obj = MLModel.objects.get(id=model_id)

    if not model_obj.parent_file.file_owner == request.user:
        messages.error(request, "Unable to delete model - Invalid Permissions")
        return HttpResponseRedirect('manage_models.html')

    model_obj.delete()
    messages.success(request, "Model deleted successfully")
    return HttpResponseRedirect('/easyml/manage/models')


def rename_model(request):
    if "GET" == request.method:
        return render(request, "manage_data.html", {})

    model_id = request.POST.get('model_id')
    new_name = request.POST.get('display_name')

    if not (request.user == MLModel.objects.get(id=model_id).parent_file.file_owner):
        return HttpResponseRedirect('/easyml/manage/models')

    if MLModel.objects.filter(parent_file__file_owner=request.user, display_name=new_name).count() > 0:
        messages.error(request, "A model with that name already exists")
        return HttpResponseRedirect('/easyml/manage/models')

    model_obj = MLModel.objects.get(id=model_id)
    model_obj.display_name = new_name
    model_obj.save()

    messages.success(request, "Model successfully renamed")
    return HttpResponseRedirect('/easyml/manage/models')

def select_csv(request, purpose=None):
    if not purpose:
        return HttpResponseRedirect('/easyml/')

    context = {}
    valid_files = CsvFile.objects.filter(file_owner=request.user)
    if not valid_files:
        valid_files = []

    context['valid_files'] = valid_files
    if purpose == 'train':
        context['title'] = "Select file to use for training"
        context['form_action'] = "select_columns_and_alg"
    else:
        context['title'] = "Select file to use for model input"
        context['form_action'] = "select_columns_and_model"

    return render(request, 'select_csv.html', context=context)

def select_columns_and_alg(request):
    if "GET" == request.method:
        return render(request, "home.html", {})

    context = {}
    file_id = int(request.POST.get('file_id'))
    if request.user != CsvFile.objects.get(id=file_id).file_owner:
        return HttpResponseRedirect('/easyml/train/setup/select-csv')

    headers = CsvFileData.objects.filter(parent_file_id=file_id)\
        .order_by('column_num')\
        .values_list('column_header', flat=True)\
        .distinct()

    context['headers'] = headers
    context['file_id'] = file_id
    context['algorithms'] = get_alg_lst()

    file_data = CsvFileData.objects.filter(parent_file_id=file_id).order_by('column_num')

    if len(set(file_data.values_list('column_num', flat=True))) <= PLOT_FEATURE_CAP:
        f = matplotlib.figure.Figure()
        buf = io.BytesIO()

        data_df = get_dataframe(file_data)
        pd.plotting.scatter_matrix(data_df)
        plt.gcf().subplots_adjust(bottom=0.15)

        plt.savefig(buf, format='png')
        plt.close(f)

        context['graphic'] = base64.b64encode(buf.getvalue()).decode('ascii')

    return render(request, 'select_columns_and_alg.html', context=context)

def select_columns_and_model(request):
    if "GET" == request.method:
        return render(request, "home.html", {})

    context = {}
    file_id = int(request.POST.get('file_id'))
    if request.user != CsvFile.objects.get(id=file_id).file_owner:
        return HttpResponseRedirect('/easyml/train/setup/select-csv')

    headers = CsvFileData.objects.filter(parent_file_id=file_id)\
        .order_by('column_num')\
        .values_list('column_header', flat=True)\
        .distinct()

    context['headers'] = headers
    context['file_id'] = file_id

    valid_files = get_user_files(request.user)
    valid_models = get_user_models(request.user)
    context['valid_models'] = valid_models
    context['valid_files'] = valid_files

    return render(request, 'select_columns_and_model.html', context=context)

def create_data(request):
    if "GET" == request.method:
        return render(request, "home.html", {})

    designation_map = {
        'ignore': COLUMN_TYPE.IGNORE,
        'input': COLUMN_TYPE.INPUT,
        'target': COLUMN_TYPE.TARGET
    }

    file_id = int(request.POST.get('file_id'))
    alg_id = int(request.POST.get('algorithm'))
    header_map = {}
    file_headers = CsvFileData.objects.filter(parent_file_id=file_id).values_list('column_header', flat=True).distinct()
    for head in file_headers:
        header_map[head] = designation_map.get(request.POST.get(head), None)

    parameters = {key: request.POST.get(key, None) for key in ALGORITHM_PARAM_MAP[alg_id]}

    error_context = request.POST.dict()
    error_context['headers'] = header_map.keys()
    error_context['algorithms'] = get_alg_lst()

    try:
        set_column_types(file_id, header_map)
    except Exception as e:
        messages.error(request, str(e))
        return render(request, 'select_columns_and_alg.html', context=error_context)

    create_model(alg_id, file_id, parameters)

    return HttpResponseRedirect('/easyml/')

def select_model(request):
    context = {}

    context['valid_models'] = get_user_models(request.user)
    context['valid_files'] = get_user_files(request.user)

    return render(request, 'select_model.html', context=context)

def select_compare(request):
    valid_files = get_user_files(request.user)
    context = {
        'valid_files': valid_files
    }

    return render(request, 'compare_data.html', context=context)


def run_model(request):
    if request.method == 'GET':
        select_model(request)

    file_id = int(request.POST.get('file_id'))
    model_id = request.POST.get('model_select')

    designation_map = {
        'ignore': COLUMN_TYPE.IGNORE,
        'input': COLUMN_TYPE.INPUT,
        'target': COLUMN_TYPE.TARGET
    }

    header_map = {}
    ignore_keys = ['csrfmiddlewaretoken', 'file_id', 'model_select']
    for prop in request.POST:
        if prop in ignore_keys:
            continue

        header_map[prop] = designation_map[request.POST.get(prop)]

    error_context = request.POST.dict()
    error_context['headers'] = header_map.keys()
    error_context['valid_models'] = get_user_models(request.user)

    try:
        set_column_types(file_id, header_map)
    except Exception as e:
        messages.error(request, str(e))
        return render(request, 'select_columns_and_model.html', context=error_context)

    file_obj = CsvFile.objects.get(id=file_id)
    model_obj = MLModel.objects.get(id=model_id)

    filename = "{}-results".format(model_obj.display_name.replace(" ", ""))

    csv_data, file_data = run_model_predict(file_obj, model_obj)
    raw_rows = csv_data.split('\n')
    row_data = []
    for row in raw_rows:
        row_data.append(row.split(','))

    csv_name = filename
    user = request.user
    if CsvFile.objects.filter(raw_name=filename, file_owner=user).exists():
        csv_name = (filename + ' (%s)') % CsvFile.objects.filter(raw_name=filename, file_owner=user).count()

    csv_obj = CsvFile(raw_name=filename, display_name=csv_name, file_owner=user)
    csv_obj.save()

    columns = list(file_data.columns.values)

    csv_data = []
    for row_index, row in file_data.iterrows():
        for i in range(len(columns)):
            header = columns[i]
            if pd.isna(row[header]) or 'unnamed' in header.lower():
                continue
            data_obj = CsvFileData(parent_file=csv_obj)
            data_obj.data = row[header]
            data_obj.row_num = row_index
            data_obj.column_num = i
            data_obj.column_header = header
            csv_data.append(data_obj)

        if len(csv_data) > 500:
            CsvFileData.objects.bulk_create(csv_data)
            csv_data = []

    CsvFileData.objects.bulk_create(csv_data)

    pseudo_buffer = Echo()
    writer = csv.writer(pseudo_buffer)
    response = StreamingHttpResponse((writer.writerow(row) for row in row_data), content_type="text/csv")

    response['Content-Disposition'] = 'attachment; filename="{}.csv"'.format(filename)

    return response
