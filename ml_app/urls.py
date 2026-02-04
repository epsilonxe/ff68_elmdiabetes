from django.urls import path
from . import views

# This defines the URL patterns for the ml_app
urlpatterns = [
    path('', views.index_view, name='index'),
    path('predict/', views.predict_view, name='predict'),
]
