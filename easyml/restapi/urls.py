from django.urls import path
from . import views

urlpatterns = [
    path('get_file_headers/<int:file_id>', views.GetFileHeaders.as_view(), name='get_file_headers'),
    path('get_accuracy/<int:ffid>&<int:sfid>&<str:header>&<str:method>', views.GetAccuracy.as_view(), name='get_accuracy'),
]