from django.urls import path
from . import views
from django.views.generic.base import TemplateView

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('upload/', TemplateView.as_view(template_name='upload_csv.html'), name='upload'),
    path('upload/csv/', views.upload_csv, name='upload_csv'),
    path('manage/data/', views.manage_data, name='manage_data'),
    path('manage/data/delete-file/<int:file_id>', views.delete_file, name='delete_file'),
    path('manage/data/rename-file/', views.rename_file, name='rename_file'),
    path('train/setup/select-csv/', views.select_csv, name='select_csv'),
    path('train/setup/select-columns/', views.select_columns, name='select_columns'),
    path('train/setup/create-data/', views.create_data, name='create_data'),
]