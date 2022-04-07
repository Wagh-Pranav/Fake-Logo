from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name = 'admin-index'),
    path('detect', views.detect, name = 'admin-detect'),
]
