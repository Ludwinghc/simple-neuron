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
  trainer = Trainer(perceptron, dataset)
  trainer.train()
  iterationList = []
  inputIterationList = []
  trainerOuputsValues = []
  trainerErrorValues = []
  iterationValue = 0
  trainerInputValues = []
  trainerPredictedValues = 0
  inputIteration = 0
  input_vector_user = [0,0]
  if request.method == "POST":	
    x1 = float(request.POST.get("x1", 0))  # Obtiene el valor del input x1
    x2 = float(request.POST.get("x2", 0))  # Obtiene el valor del input x2
    input_vector_user = np.array([x1, x2])  #
    print(input_vector_user[0])
    print(input_vector_user[1])
    # Llamado a los metodos del trainer para mostrar los resultados
    # Entrenamiento
    trainerOuputsValues, trainerErrorValues, iterationValue = trainer.train()
    # Evaluacion
    trainerPredictedValues, inputIteration = trainer.evaluate(input_vector_user)
    iterationList = list(range(iterationValue))  # Lista de 0 a iterationValue-1
    inputIterationList = list(range(inputIteration))
    
    
  return render(
    request,
      'pages/andGate.html',
      {
        "dataInput": dataset.input_vector.tolist(),
        "iteracion" : list(range(iterationValue)),
        "inputIteration" : list(range(inputIteration)),
        "Outputs" : [out.tolist() if isinstance(out, np.ndarray) else out for out in trainerOuputsValues],
        "Error" : trainerErrorValues, 
        "x1": input_vector_user[0],
        "x2": input_vector_user[1],
        "PredictedValues": trainerPredictedValues,
        })