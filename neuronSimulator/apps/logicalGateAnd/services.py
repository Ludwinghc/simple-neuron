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


# Definicion de la clase de entrenamiento
class Trainer:
    # Definicion de los atributos de la clase
    def __init__(self, perceptron, dataset, iterations=12):
        self.perceptron = perceptron
        self.input_vector, self.expected_vector = dataset.get_data()
        self.iterations = iterations
    # metodo para hacer la ciclo de entrenamiento
    def train(self):
        outputResults = []
        errorResults = []
        for _ in range(self.iterations):
            print(f"Iteración #: {_}")
            for i in range(len(self.input_vector)):
                output = self.perceptron.predict(self.input_vector[i])
                error = self.expected_vector[i] - output
                self.perceptron.update_weights(self.input_vector[i], error)
                outputResults.append(output)
                errorResults.append(error)
        
        return outputResults, errorResults, self.iterations
    # Definicion para mostrar los datos
    def evaluate(self):
        Input = []
        predictions = []
        for i in range(len(self.input_vector)):
            output = self.perceptron.predict(self.input_vector[i])
            Input.append(self.input_vector[i])
            predictions.append(output)
        return Input, predictions, len(self.input_vector)


