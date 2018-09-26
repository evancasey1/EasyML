from django.urls import path
from . import views
from django.views.generic.base import TemplateView

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('upload/', TemplateView.as_view(template_name='upload_csv.html'), name='upload'),
    path('upload/csv/', views.upload_csv, name='upload_csv'),
    path('manage/data/', views.manage_data, name='manage_data'),
    path('manage/data/delete_file/(?P<file_id>\w+)', views.delete_file, name='delete_file'),
    path('manage/data/rename_file/(?P<file_id>\w+)', views.rename_file, name='rename_file'),
]