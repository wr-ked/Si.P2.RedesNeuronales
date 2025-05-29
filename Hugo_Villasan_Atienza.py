# Hugo Villasan Atienza. Sistemas Inteligentes. P2. Redes Neuronales.

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay



def cargar_y_preprocesar_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normaliza los pixeles de 0-255 a 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    
    #Convierte etiquetas a one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

def MLP(input_shape, ocultas = 32, activ = "sigmoid", ep = 10, bs = 32, val_split = 0.1, early_stopping=False):
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
    
    # EarlyStopping
    callbacks = []
    if early_stopping:
        cb = EarlyStopping(
            monitor="val_loss",
            patience=3,
            min_delta=0.001,
            restore_best_weights=True
        )
        callbacks = [cb]
    
    # Entrena
    history = model.fit(
        X_train, y_train,
        batch_size = bs,
        epochs = ep,
        validation_split = val_split,
        callbacks = callbacks,
        verbose = 1
    )
    
    # Evalua
    score = model.evaluate(X_test, y_test, verbose = 0)
    print ("Test accuracy:",score[1])
    print ("Test loss:",score[0])
    
    return model, history, X_test, y_test

    

def ajuste_epochs(input_shape, repeticiones = 5):
    epochs_list = [5, 10, 15, 20, 30]
    
    for ep in epochs_list:
        print(f"\n--- Ajuste con {ep} epochs ({repeticiones} repeticiones) ---")
        histories = []
        
        for i in range(repeticiones):
            print(f"  → Repetición {i+1}/{repeticiones}")
            _, history = MLP(
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
    
        
def ajuste_validation_split(input_shape, ep_optimo, repeticiones = 5):
    splits = [0.05, 0.1, 0.2]
    
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
    
def ajuste_ES(input_shape, ep_optimo = 20, val_split_optimo = 0.2, repeticiones = 5):
    
    configuraciones = [
        {"patience": 2, "min_delta": 0.001},
        {"patience": 3, "min_delta": 0.001},
        {"patience": 5, "min_delta": 0.0005},
    ]
    

    medias_accuracy = []
    medias_epocas = []

    
    for cfg in configuraciones:
        print(f"\n--- Probar ES con patience={cfg['patience']} / min_delta={cfg['min_delta']} ---")
        acc_total = 0
        ep_total = 0
        
        for i in range(repeticiones):
            model, history, X_test, y_test = MLP(
                input_shape=input_shape,
                ocultas=[32],
                activ=["sigmoid"],
                ep=ep_optimo,
                bs=32,
                val_split=val_split_optimo
                )
            
            # Número de épocas reales entrenadas
            ep_entrenadas = len(history.history["validation"])
            ep_total += ep_entrenadas
            
            # Evalúa en test
            _, acc_test = model.evaluate(X_test, y_test, verbose=0)
            acc_total += acc_test
            
        medias_accuracy.append(acc_total / repeticiones)
        medias_epocas.append(ep_total / repeticiones)


def ajuste_batch_size(input_shape, ep_optimo=20, val_split_optimo=0.1, repes=5):
    batch_sizes = [16, 32, 64, 128]
    
    medias_accuracy = []
    medias_tiempo = []
    confusion_matrices = {}

    for bs in batch_sizes:
        print(f"\n--- Entrenando con batch_size={bs} ---")
        acc_total = 0
        tiempo_total = 0
        all_y_preds = []
        all_y_trues = []
        
        for i in range(repes):
            print(f"  → Repetición {i+1}/{repes}")
            start_time = time.time() # Medimos el tiempo que tarda el modelo en entrenarse
            model, _, X_test, y_test  = MLP(
                input_shape=input_shape,
                ocultas=[32],
                activ=["sigmoid"],
                ep=ep_optimo,
                bs=bs,
                val_split=val_split_optimo
                )
            end_time = time.time()
            tiempo_total += (end_time - start_time)
            
            # Evalúa test acc
            _, acc_test = model.evaluate(X_test, y_test, verbose=0)
            acc_total += acc_test
            
            # Guarda predicciones para matriz confusión
            y_pred = np.argmax(model.predict(X_test), axis=1)
            y_true = np.argmax(y_test, axis=1)
            all_y_preds.extend(y_pred)
            all_y_trues.extend(y_true)
            
        medias_accuracy.append(acc_total / repes)
        medias_tiempo.append(tiempo_total / repes)

        
        # Calcula matriz de confusión promedio
        cm = confusion_matrix(all_y_trues, all_y_preds)
        confusion_matrices[bs] = cm
        
    dibuja_batch_size_resultados(batch_sizes, medias_tiempo, medias_accuracy, ep_optimo, repes)
    muestra_matrices_confusion(batch_sizes, confusion_matrices)
    
    
#def ajuste_act_function(input_shape, ep_optimo=20, val_split_optimo=0.1, bs_optimo=32, repes=5):
    
# def ajuste_No_capas():
#     
# def CNN():
    
            
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
    plt.legend(loc='lower right')
    plt.grid(axis='y')

    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica de barras guardada en {ruta}")
    plt.show()    
    
def dibuja_batch_size_resultados(batch_sizes, medias_tiempo, medias_accuracy, ep_optimo, repes, nombre_archivo="ajuste_batch_size"):

    x = np.arange(len(batch_sizes))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tiempo = 'skyblue'
    color_acc = 'salmon'


    ax1.set_xlabel('Batch size')
    ax1.set_ylabel('Tiempo (s)', color=color_tiempo)
    ax1.bar(x - 0.2, medias_tiempo, width=0.4, color=color_tiempo, label='Tiempo (s)')
    ax1.tick_params(axis='y', labelcolor=color_tiempo)
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.grid(axis='y')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ax2.bar(x + 0.2, medias_accuracy, width=0.4, color=color_acc, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title(f"Ajuste de batch_size (ép. máx={ep_optimo}, {repes} repeticiones)")

    # Guardar y mostrar
    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre_archivo}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica guardada en {ruta}")
    plt.show()
    plt.close()


    
def muestra_matrices_confusion(batch_sizes, confusion_matrices):
    for bs in batch_sizes:
        cm = confusion_matrices[bs]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - Batch Size {bs}")
        plt.grid(False)
        os.makedirs("graficas", exist_ok=True)
        plt.savefig(f"graficas/conf_matrix_bs_{bs}.png")
        plt.show()



if __name__ == "__main__":
     
    # Tarea A. Definir, utilizar y evaluar un MLP con Keras
    #MLP()
    
    # Tarea B. Ajustar el valor de los parámetros epochs y validation_split
    
    #ajuste_epochs()
    #ajuste_validation_split(input_shape=(32, 32, 3), ep_optimo=20, repeticiones=5)

    # Tarea C. Ajustar el valor del parámetro batch_size
    ajuste_batch_size(input_shape=(32, 32, 3), ep_optimo=20, val_split_optimo=0.2, repes=5)

    
    # Tarea D. Probar diferentes funciones de activación
    
    # Tarea E. Ajustar el número de neuronas por capa
    
    # Tarea F. Optimizar un MLP de dos o más capas
    