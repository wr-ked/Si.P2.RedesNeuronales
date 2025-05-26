# Hugo Villasan Atienza. Sistemas Inteligentes. P2. Redes Neuronales.
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_y_preprocesar_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normaliza los pixeles de 0-255 a 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    
    #Convierte etiquetas a one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

def MLP(input_shape, ocultas = [32], activ = ["sigmoid"], ep = 10, bs = 32, val_split = 0.1):
    model = Sequential (name = "MLP")
    model.add(Flatten(input_shape = input_shape))
    
    # Añade capas ocultas
    for n_neuronas, func_act in zip(ocultas, activ):
        model.add(Dense(n_neuronas))
        model.add(Activation(func_act))
        
    # Capa de salida
        
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    #Compila
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = Adam(),
        metrics = ["accuracy"]
    )
    
    model.summary()
    
    # Carga los datos
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10()
    
    # Entrena
    history = model.fit(
        X_train, y_train,
        batch_size = bs,
        epochs = ep,
        validation_split = val_split,
        verbose = 1
    )
    
    # Evalua
    score = model.evaluate(X_test, y_test, verbose = 0)
    print ("Test accuracy:",score[1])
    print ("Test loss:",score[1])
    
    return model, history
    

def ajuste_epochs(input_shape, epochs_list = [5, 10, 15, 20, 30], repeticiones = 5):
    
    for ep in epochs_list:
        print(f"\n--- Ajuste con {ep} epochs ({repeticiones} repeticiones) ---")
        histories = []
        
        for i in range(repeticiones):
            print(f"  → Repetición {i+1}/{repeticiones}")
            model, history = MLP(
                input_shape=input_shape,
                ocultas=[32],
                activ=["sigmoid"],
                ep=ep,
                bs=32,
                val_split=0.1
            )
            histories.append(history.history)
            
        # Promedio de las curvas
        history_keys = histories[0].keys()
        history_avg = {key: np.zeros(ep) for key in history_keys}

        for h in histories:
            for key in history_keys:
                history_avg[key] += np.array(h[key])
        for key in history_keys:
            history_avg[key] /= repeticiones
            
        # Dibuja la grafica                
        dibuja_grafica_de_precision(history_avg, nombre=f"epochs_{ep}")
        
def dibuja_grafica_de_precision(history_dict, nombre="grafica"):
    # Asegura que existe carpeta de salida
    os.makedirs("graficas", exist_ok=True)
    
    # Asegura empezar siempre en un canva vacio
    plt.figure()
    
    # Curvas acc
    plt.plot(history_dict["accuracy"], label="Train acc")
    plt.plot(history_dict["val_accuracy"], label="Val acc")
    
    # Estetica
    plt.legend()
    plt.grid()
    plt.title("Evolución del entrenamiento")
    
    # Guarda la grafica
    ruta = f"graficas/vacc_{nombre}.png"
    plt.savefig(ruta)
    
    # Asegura empezar siempre en un canva vacio
    plt.figure()
    
    # Curvas loss
    plt.plot(history_dict["loss"], label="Train loss")
    plt.plot(history_dict["val_loss"], label="Val loss")
    
    # Estetica
    plt.legend()
    plt.grid()
    plt.title("Evolución del entrenamiento")
    
    # Guarda la grafica
    ruta = f"graficas/loss_{nombre}.png"
    plt.savefig(ruta)
        
def ajuste_validation_split(input_shape, ep_optimo, splits = [0.05, 0.1, 0.2], repeticiones = 5):
    val_accs = []
    test_accs = []
    for val_split in splits:
        acc_val_total = 0
        acc_test_total = 0
        print(f"\n--- Entrenando con validation_split={val_split} ---")
        for i in range (repeticiones):
            print(f"  → Repetición {i+1}/{repeticiones}")
            model, history = MLP(
                input_shape=input_shape,
                ocultas=[32],
                activ=["sigmoid"],
                ep=ep_optimo,
                bs=32,
                val_split=val_split
            )
            
            # acc_val y acc_test de cada vuelta
            acc_val_total += history.history["val_accuracy"][-1]
            _, acc_test = model.evaluate(*cargar_y_preprocesar_cifar10()[2:], verbose=0)
            acc_test_total += acc_test
            
        # Promedio
        val_accs.append(acc_val_total / repeticiones)
        test_accs.append(acc_test_total / repeticiones)
        
    
    # Representacion grafica
    dibuja_barras_accuracy_splits(splits, val_accs, test_accs, ep_optimo, repeticiones)
    
def dibuja_barras_accuracy_splits(splits, val_accs, test_accs, ep_optimo, repeticiones, nombre="comparacion_valsplit"):
    x = np.arange(len(splits))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, val_accs, width, label="Val Accuracy", color="skyblue")
    plt.bar(x + width/2, test_accs, width, label="Test Accuracy", color="salmon")

    plt.xticks(x, [f"{int(s*100)}%" for s in splits])
    plt.xlabel("validation_split")
    plt.ylabel("Accuracy")
    plt.title(f"Comparación de accuracies\n(épocas={ep_optimo}, repeticiones={repeticiones})")
    plt.legend()
    plt.grid(axis='y')

    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre}.png"	
    plt.savefig(ruta)
    print(f"[✔] Gráfica de barras guardada en {ruta}")
    plt.show()
            
        



if __name__ == "__main__":
     
    # Tarea A. Definir, utilizar y evaluar un MLP con Keras
    #MLP()
    
    # Tarea B. Ajustar el valor de los parámetros epochs y validation_split
    
    #ajuste_epochs()
    ajuste_validation_split(input_shape=(32, 32, 3), ep_optimo=20, repeticiones=5)

    # Tarea C. Ajustar el valor del parámetro batch_size
    
    # Tarea D. Probar diferentes funciones de activación
    
    # Tarea E. Ajustar el número de neuronas por capa
    
    # Tarea F. Optimizar un MLP de dos o más capas
    