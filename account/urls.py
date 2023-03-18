from django.urls import path, include
from . import views

urlpatterns = [
    path('login', views.login, name='account-login'),
    path('signup', views.signup, name='account-signup'),
    path('logout', views.logout, name='account-logout'),
]
