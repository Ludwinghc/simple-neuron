from django.urls import path
from . import views

urlpatterns = [ 
    path("andGate/", views.andGate, name="andGate"),
]