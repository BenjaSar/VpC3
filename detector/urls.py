"""
URLs de la app detector
"""

from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.detect, name='detect'),
]
