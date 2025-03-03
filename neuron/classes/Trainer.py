# trainer.py
# from Data import Data
# from Perceptron import Perceptron

# Definicion de la clase de entrenamiento
class Trainer:
    # Definicion de los atributos de la clase
    def __init__(self, perceptron, dataset, iterations=5):
        self.perceptron = perceptron
        self.input_vector, self.expected_vector = dataset.get_data()
        self.iterations = iterations
    # metodo para hacer la ciclo de entrenamiento
    def train(self):
        for _ in range(self.iterations):
            print(f"Iteración #: {_}")
            for i in range(len(self.input_vector)):
                output = self.perceptron.predict(self.input_vector[i])
                error = self.expected_vector[i] - output
                self.perceptron.update_weights(self.input_vector[i], error)
                print(f"Salida: {output} con un error del : {error}")
    # Definicion para mostrar los datos
    def evaluate(self):
        for i in range(len(self.input_vector)):
            output = self.perceptron.predict(self.input_vector[i])
            print(f"Entrada: {self.input_vector[i]}, salida predicha: {output}")


