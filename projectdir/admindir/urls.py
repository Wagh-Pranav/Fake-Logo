from django.urls import path, include
from . import views

urlpatterns = [
    path('train', views.index, name = 'admin-index'),
]
