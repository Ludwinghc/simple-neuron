# perceptron.py
import numpy as np

# Definición de clase para el perceptron
class Perceptron:
    # Definición de los atributos de la clase
    def __init__(self, input_size, alpha=0.1):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.alpha = alpha
    
    # Metodo para la ejecutar la función de activacion
    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    # Metodo para la ejecutar la prediccion del perceptron
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)
    #  Metodo para actualizar los pesos
    def update_weights(self, inputs, error):
        self.weights += self.alpha * error * inputs
        self.bias += self.alpha * error
