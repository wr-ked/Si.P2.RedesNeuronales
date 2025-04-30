from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np

def cargar_y_preprocesar_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    #Normaliza los pixeles de 0-255 a 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    
    #Convierte etiquetas a one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

def probar_MLP(input_shape, ocultas = [32], activ = ["sigmoid"], ep = 10, bs = 32, val_split = 0.1):
    model = Sequential (name = "MLP_A")
    model.add(Flatten(input_shape = input_shape))
    
    #Añade capas ocultas
    for n_neuronas, func_act in zip(ocultas, activ):
        model.add(Dense(n_neuronas))
        model.add(Activation(func_act))
        
    #Capa de salida
        
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    #Compila
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = Adam(),
        metrics = ["accuracy"]
    )
    
    model.summary()
    
    #Carga los datos
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10()
    
    #Entrena
    history = model.fit(
        X_train, y_train,
        batch_size = bs,
        epochs = ep,
        validation_split = val_split,
        verbose = 1
    )
    
    #Evalua
    score = model.evaluate(X_test, y_test, verbose = 0)
    print ("Test accuracy:",score[1])
    print ("Test loss:",score[1])
    
    return model, history


if __name__ == "__main__":
    model, history = probar_MLP(
        input_shape = (32, 32, 3), # tamaño de la imagen de CIFAR10
        ocultas = [32],
        activ = ["sigmoid"],
        ep = 10,
        bs = 32,
        val_split = 0.1        
    )
    