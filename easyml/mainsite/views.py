import pandas as pd
import traceback

from .models import CsvFile, CsvFileData
from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse_lazy, reverse
from django.views import generic
from .forms import CustomUserCreationForm

# Create your views here.
def index(request):
    return HttpResponse("This is the EasyBakeML index.")

class SignUp(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'

def upload_csv(request):
    data = {}
    if "GET" == request.method:
        return render(request, "upload_csv.html", data)

    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'File is not CSV type')
            return HttpResponseRedirect(reverse("upload_csv"))
        # if file is too large, return
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)" % (csv_file.size / (1000 * 1000),))
            return HttpResponseRedirect(reverse("upload_csv"))

        user = request.user
        if CsvFile.objects.filter(name=csv_file.name, file_owner=user).exists():
            messages.error(request, 'A File with this name already exists')
            return HttpResponseRedirect(reverse("upload_csv"))

        csv_obj = CsvFile(name=csv_file.name, file_owner=user)
        csv_obj.save()

        file_data = csv_file.read().decode("utf-8")

        csv_data = []
        lines = file_data.split("\n")
        row_num = 0
        for line in lines:
            data_lst = line.split(',')
            for i in range(len(data_lst)):
                try:
                    data_obj = CsvFileData(parent_file=csv_obj)
                    data_obj.data = float(data_lst[i])
                    data_obj.row_num = row_num
                    data_obj.column_num = i
                    csv_data.append(data_obj)
                except:
                    pass

            row_num += 1

        CsvFileData.objects.bulk_create(csv_data)

    except Exception as e:
        print(traceback.format_exc(e))
        messages.error(request, "Unable to upload file. " + repr(e))

    return HttpResponseRedirect("/")
