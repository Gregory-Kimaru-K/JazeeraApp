from django.urls import path
from .views import predict_view, home

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_view, name='predict'),
]