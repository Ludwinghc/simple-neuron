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
  iterationList = []
  inputIterationList = []
  trainerOuputsValues = []
  trainerErrorValues = []
  iterationValue = 0
  trainerInputValues = []
  trainerPredictedValues = []
  inputIteration = 0
  if request.method == "POST":	

    
    
    # Llamado a los metodos del trainer para mostrar los resultados
    trainerOuputsValues, trainerErrorValues, iterationValue = trainer.train()
    trainerInputValues, trainerPredictedValues, inputIteration = trainer.evaluate()
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
        "Inputs" : [inp.tolist() for inp in trainerInputValues],
        "PredictedValues" : [pred.tolist() if isinstance(pred, np.ndarray) else pred for pred in trainerPredictedValues],
        })