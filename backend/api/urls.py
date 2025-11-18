from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_excel),
    path('query/', views.query_area),                          # GET: health check
]
