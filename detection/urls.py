from django.urls import path, include
from . import views

urlpatterns = [
    path('augment', views.augment, name = 'admin-augment'),
    path('test', views.test, name = 'admin-test'),
    path('upload', views.upload, name = 'upload'),
    path('detect/<int:pk>', views.detect, name = 'detect'),
]
