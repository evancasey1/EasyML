import django
import datetime

from django.db import models
from django.contrib.auth.models import AbstractUser
from picklefield.fields import PickledObjectField

class CustomUser(AbstractUser):
    # add additional fields in here

    def __str__(self):
        return self.email

class CsvFile(models.Model):

    file_owner = models.ForeignKey(
        'CustomUser',
        related_name="file_owner",
        null=False,
        blank=False,
        on_delete=models.CASCADE)
    raw_name = models.CharField(max_length=255)
    display_name = models.CharField(max_length=255, unique=True, null=False, blank=False)
    created = models.DateTimeField(auto_now_add=True, blank=True)

class CsvFileData(models.Model):

    parent_file = models.ForeignKey(
        'CsvFile',
        related_name="parent_file",
        null=False,
        blank=False,
        on_delete=models.CASCADE)
    column_header = models.CharField(max_length=255)
    data = models.FloatField(null=True, blank=True)
    placeholder = models.TextField(null=True, blank=True)
    row_num = models.IntegerField(null=False, blank=False)
    column_num = models.IntegerField(null=False, blank=False)
    type = models.IntegerField(null=True, blank=True)

class MLModel(models.Model):
    type = models.CharField(max_length=255, blank=False, null=False)
    type_num = models.IntegerField(blank=False, null=False)
    data = PickledObjectField()
    created_at = models.DateTimeField(auto_now_add=True)
    parent_file = models.ForeignKey(
        'CsvFile',
        related_name="model_parent",
        null=False,
        blank=False,
        on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    display_name = models.CharField(max_length=255)
    parameters = models.TextField(blank=False, null=False)
    accuracy = models.FloatField(null=True, blank=True)
    accuracy_type = models.CharField(max_length=255, null=True, blank=True)
