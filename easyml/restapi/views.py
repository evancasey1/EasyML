from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import (
    PermissionDenied,
    APIException,
)
from rest_framework import viewsets, mixins
from django.http import JsonResponse, Http404
from django.http import HttpResponseForbidden

from helpers.util import *
from mainsite.models import *

class BaseUserView(APIView):
    permission_classes = [IsAuthenticated]

class GetFileHeaders(BaseUserView):

    def get(self, request, file_id=None):
        if not file_id:
            raise APIException("file_id not provided")

        file_obj = CsvFile.objects.filter(id=file_id)
        if not file_obj:
            raise Http404

        file_obj = file_obj.first()
        if file_obj.file_owner != request.user:
            raise HttpResponseForbidden

        data_raw = CsvFileData.objects.filter(parent_file_id=file_id)\
            .values_list('column_header', flat=True)\
            .distinct()

        header_data = {'headers': sorted(list(data_raw))}

        return JsonResponse(header_data)

class GetAccuracy(BaseUserView):

    def get(self, request, ffid=None, sfid=None, header=None, method=None):
        if not ffid:
            raise APIException("First file_id not provided")

        if not sfid:
            raise APIException("Second file_id not provided")

        if not header:
            raise APIException("Header not provided")

        if not method:
            raise APIException("Method not provided")

        ffid = int(ffid)
        sfid = int(sfid)

        first_data_raw = CsvFileData.objects.filter(parent_file_id=ffid,
                                                    column_header=header) \
            .order_by('row_num')

        second_data_raw = CsvFileData.objects.filter(parent_file_id=sfid,
                                                     column_header=header) \
            .order_by('row_num')

        itos_map1 = get_itos_map(ffid)
        itos_map2 = get_itos_map(sfid)

        first_data = []
        for dpoint in first_data_raw:
            if dpoint.column_header in itos_map1:
                first_data.append(itos_map1[dpoint.column_header][dpoint.data])
            else:
                first_data.append(dpoint.data)

        second_data = []
        for dpoint in second_data_raw:
            if dpoint.column_header in itos_map2:
                second_data.append(itos_map2[dpoint.column_header][dpoint.data])
            else:
                second_data.append(dpoint.data)

        if len(first_data) != len(second_data):
            messages = []
            if len(second_data) == 0:
                fname = CsvFile.objects.get(id=sfid).display_name
                messages.append('Header "{}" does not exist in file "{}"'.format(header, fname))

            messages.append('Length of the two files is not identical. {} vs {} rows.'
                            .format(len(first_data), len(second_data)))

            return JsonResponse({
                'status_code': 500,
                'messages': messages
            })

        if 'accuracy' in method.lower():
            acc = str(get_match_acc(first_data, second_data)) + "%"
            accuracy_type = 'Accuracy'
            method_sm = 'Accuracy'
        else:
            acc = get_r2(first_data, second_data)
            accuracy_type = 'R^2'
            method_sm = 'Correlation'

        context = {
            'status_code': 200,
            'accuracy': acc,
            'accuracy_type': accuracy_type,
            'method_sm': method_sm,
            'num_rows': len(first_data),
        }

        return JsonResponse(context)
