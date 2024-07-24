from django.urls import path
from .views import home, predict_view, visualize_view

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_view, name='predict'),
    path('visualize/', visualize_view, name='visualize'),
]