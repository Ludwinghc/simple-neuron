# Llamado a clases
from classes.Data import Data
from classes.Perceptron import Perceptron
from classes.Trainer import Trainer

if __name__ == "__main__":
    # Nuevo llamado a la instancia  de la data
    dataset = Data()
    # Nuevo llamado a la instancia del perpectron 
    perceptron = Perceptron(input_size=2, alpha=0.1)
    # Nuevo llamado a la instancia del trainer
    trainer = Trainer(perceptron, dataset)
    # Llamado a los metodos del trainer para mostrar los resultados
    trainer.train()
    trainer.evaluate()
