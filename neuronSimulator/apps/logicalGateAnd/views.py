from django.shortcuts import render, HttpResponse, redirect
# Llamado a funciones
from .utils import mi_funcion

# Create your views here.
def andGate(request):
  message = ""
  if request.method == "POST":	
    message = mi_funcion()
  return render(request, 'pages/andGate.html',{"mensaje":message})