from django.shortcuts import render, HttpResponse, redirect

# Llamado a las clases
from .services import Data, Perceptron, Trainer

import numpy as np
# Create your views here.
def andGate(request):
  # Nuevo llamado a la instancia  de la data
  dataset = Data()
  # Nuevo llamado a la instancia del perpectron 
  perceptron = Perceptron(input_size=2, alpha=0.1)
  # Nuevo llamado a la instancia del trainer
  trainerPredictedValues = []
  inputIteration = []
  iterations = 0

  if request.method == "POST":	
    # Captura de datos del formulario
    iterations = int(request.POST.get('iterations'))
    input1 = int(request.POST.get('input1'))
    input2 = int(request.POST.get('input2'))
    inputIteration = np.array([[input1, input2]])
    trainer = Trainer(perceptron, dataset, iterations=iterations)
    # Llamado a los metodos del trainer para mostrar los resultados
    trainer.train()
    trainerPredictedValues = trainer.evaluate(input=inputIteration)
    
    
  return render(
    request,
      'pages/andGate.html',
      {
        "dataInput": dataset.input_vector.tolist(),
        "Inputs" : [inp.tolist() for inp in inputIteration],
        "PredictedValues" : [pred.tolist() if isinstance(pred, np.ndarray) else pred for pred in trainerPredictedValues],
        })