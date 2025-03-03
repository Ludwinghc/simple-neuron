import numpy as np

# Clase para la data para entrenar el perceptron
class Data:
    # Definición de los atributos de la clase
    def __init__(self):
        #  Definición del vector de entrada
        self.input_vector = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        #  Definicion del vector esperado para la salida
        self.expected_vector = np.array([0, 0, 0, 1])
    
    # Metodo para traer la data
    def get_data(self):
        return self.input_vector, self.expected_vector
