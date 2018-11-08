from django.urls import path
from . import views
from django.views.generic.base import TemplateView

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('upload/', TemplateView.as_view(template_name='upload_csv.html'), name='upload'),
    path('upload/csv/', views.upload_csv, name='upload_csv'),
    path('upload/csv/<str:next>/', views.upload_csv, name='upload_csv'),
    path('manage/data/', views.manage_data, name='manage_data'),
    path('manage/models/', views.manage_models, name='manage_models'),
    path('manage/data/delete-file/<int:file_id>', views.delete_file, name='delete_file'),
    path('manage/data/rename-file/', views.rename_file, name='rename_file'),
    path('manage/models/delete-model/<int:model_id>', views.delete_model, name='delete_model'),
    path('manage/models/rename-model/', views.rename_model, name='rename_model'),
    path('train/setup/select-csv/<str:purpose>', views.select_csv, name='select_csv'),
    path('predict/setup/select-csv/<str:purpose>', views.select_csv, name='select_csv'),
    path('compare/setup/', views.select_compare, name='select_compare'),
    path('train/setup/select-columns/', views.select_columns_and_alg, name='select_columns_and_alg'),
    path('predict/setup/select-columns/', views.select_columns_and_model, name='select_columns_and_model'),
    path('train/setup/create-data/', views.create_data, name='create_data'),
    path('predict/select-model/', views.select_model, name='select_model'),
    path('predict/run-model/', views.run_model, name='run_model'),
]