# Libreria para manejo de matrices y arreglos
import numpy as np

# Declaración de variables
# Vector de entradas
inputVector = np.array([[0,0], [0,1], [1,0], [1,1]])
# Vector de la salida esperada para la compuerta logica AND
expectedVector = np.array([0,0,0,1])

# Iniciacion de pesos y parámetros
# Pesos
weights = np.random.rand(2)
# Sesgo
bias = np.random.rand(1)
# Learning percentage
alpha = 0.1

# Función de activacion
def activation(sum):
  return 1 if sum >= 0 else 0

# Entrenamiento del perceptron
for iteration in range(10):
  # Ciclo para el largo del vector de entradas
  for i in range(len(inputVector)):
    # Funcion del perpectron simple
    weightedSum = np.dot(inputVector[i], weights) + bias
    # Salida de la funcion del perceptron
    output = activation(weightedSum)
    # evaluacion de la salida
    error = expectedVector[i] - output
    # Ajuste de pesos
    weights += alpha * error * inputVector[i]
    # Ajuste del sesgo
    bias += alpha * error

# Verificación del resultado
for i in range(len(inputVector)):
  weightedSum = np.dot(inputVector[i], weights) + bias
  finalOutput = activation(weightedSum)
  print(f"Entrada: {inputVector[i]}, salida predicha: {finalOutput}")