# Generated by Django 2.1.1 on 2018-10-04 18:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0010_mlmodel_parent_file'),
    ]

    operations = [
        migrations.RenameField(
            model_name='mlmodel',
            old_name='model_data',
            new_name='data',
        ),
    ]