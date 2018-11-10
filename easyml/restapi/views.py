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

        headers = list(CsvFileData.objects.filter(parent_file_id=file_id)
                       .values_list('column_header', flat=True).distinct())

        header_data = {'headers': headers}

        return JsonResponse(header_data)
