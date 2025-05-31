# Hugo Villasan Atienza. Sistemas Inteligentes. P2. Redes Neuronales.

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
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

def MLP(input_shape, ocultas = 32, activ = "sigmoid", ep = 10, bs = 32, val_split = 0.1):
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
    cb = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
        mode="max"  
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
    
def ajuste_ES(input_shape, ep_max = 50, val_split_optimo = 0.1, repeticiones = 5):
    
    configuraciones = [
        {"patience": 2, "min_delta": 0.001},  # Modelo exigente
        {"patience": 3, "min_delta": 0.001},  # Modelo equilibrado
        {"patience": 5, "min_delta": 0.0005}, # Modelo paciente
    ]
    

    medias_accuracy = []
    medias_epocas = []

    
    for cfg in configuraciones:
        print(f"\n--- Probar ES con patience={cfg['patience']} / min_delta={cfg['min_delta']} ---")
        acc_total = 0
        ep_total = 0
        
        callback_es = EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["patience"],
            min_delta=cfg["min_delta"],
            restore_best_weights=True,
            mode = "max"
            
        )

        
        for _ in range(repeticiones):
            model, history, X_test, y_test = MLP(
                input_shape=input_shape,
                ocultas=[32],
                activ=["sigmoid"],
                ep=ep_max,
                bs=32,
                val_split=val_split_optimo,
                early_stopping=[callback_es]
                )
            
            # Número de épocas reales entrenadas
            ep_entrenadas = len(history.history["val_loss"])
            ep_total += ep_entrenadas
            
            # Evalúa en test
            _, acc_test = model.evaluate(X_test, y_test, verbose=0)
            acc_total += acc_test
            
        medias_accuracy.append(acc_total / repeticiones)
        medias_epocas.append(ep_total / repeticiones)
        
    dibuja_resultados_ES(configuraciones, medias_epocas, medias_accuracy, ep_max, repeticiones, nombre_archivo="ajuste_ES")


def ajuste_batch_size(input_shape, ep_max=20, val_split_optimo=0.1, repes=5):
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
                ep=ep_max,
                bs=bs,
                val_split=val_split_optimo,
                early_stopping = True
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
        
    dibuja_batch_size_resultados(batch_sizes, medias_tiempo, medias_accuracy, ep_max, repes)
    muestra_matrices_confusion(batch_sizes, confusion_matrices)
    
    
def ajuste_act_function(input_shape, ep_optimo=20, val_split_optimo=0.1, bs_optimo=64, repeticiones=5):
    funciones = ["sigmoid", "relu", "tanh", "mish"]  # activaciones a comparar
    tiempos_medios = []
    accuracies_medias = []

    for activ in funciones:
        print(f"\n--- Probar activación: {activ} ---")
        tiempo_total = 0
        acc_total = 0

        for i in range(repeticiones):
            start = time.time()
            
            model, history, X_test, y_test = MLP(
                input_shape=input_shape,
                ocultas=[32],
                activ=[activ],
                ep=ep_optimo,
                bs=bs_optimo,
                val_split=val_split_optimo,
                early_stopping = True
            )

            _, acc_test = model.evaluate(X_test, y_test, verbose=0)
            end = time.time()

            tiempo_total += (end - start)
            acc_total += acc_test

        tiempos_medios.append(tiempo_total / repeticiones)
        accuracies_medias.append(acc_total / repeticiones)

    dibuja_activaciones_resultados(funciones, tiempos_medios, accuracies_medias, ep_optimo, repeticiones)
    
    
def ajuste_No_capas(input_shape, ep_optimo=20, val_split_optimo=0.2, bs_optimo=64, repeticiones=5):
    num_neuronas_lista = [16, 32, 64, 128, 256, 512]  # puedes añadir más si quieres
    tiempos_medios = []
    accuracies_medias = []

    for n in num_neuronas_lista:
        print(f"\n--- Probar con {n} neuronas ---")
        tiempo_total = 0
        acc_total = 0

        for i in range(repeticiones):
            start = time.time()

            model, history, X_test, y_test = MLP(
                input_shape=input_shape,
                ocultas=[n],
                activ=["relu"],
                ep=ep_optimo,
                bs=bs_optimo,
                val_split=val_split_optimo,
                early_stopping = True
            )

            _, acc_test = model.evaluate(X_test, y_test, verbose=0)
            end = time.time()

            tiempo_total += (end - start)
            acc_total += acc_test

        tiempos_medios.append(tiempo_total / repeticiones)
        accuracies_medias.append(acc_total / repeticiones)
        
    dibuja_NoNeuronas_resultados(num_neuronas_lista, tiempos_medios, accuracies_medias, ep_optimo, repeticiones)
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
    
def dibuja_resultados_ES(configuraciones, medias_epocas, medias_accuracy, ep_optimo, repes, nombre_archivo="ajuste_ES"):

    x = np.arange(len(configuraciones))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_epocas = 'skyblue'
    color_acc = 'salmon'

    # Etiquetas para eje x tipo: patience=2, min_delta=0.001
    etiquetas = [f"p={cfg['patience']}\nmd={cfg['min_delta']}" for cfg in configuraciones]

    ax1.set_xlabel('Configuraciones EarlyStopping')
    ax1.set_ylabel('Épocas entrenadas (media)', color=color_epocas)
    ax1.bar(x - 0.2, medias_epocas, width=0.4, color=color_epocas, label='Épocas entrenadas')
    ax1.tick_params(axis='y', labelcolor=color_epocas)
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas)
    ax1.grid(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy (media)', color=color_acc)
    ax2.bar(x + 0.2, medias_accuracy, width=0.4, color=color_acc, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title(f"Ajuste EarlyStopping (ép. máx={ep_optimo}, {repes} repeticiones)")

    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre_archivo}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica guardada en {ruta}")
    plt.show()
    plt.close()
    
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
        
        
def dibuja_activaciones_resultados(activaciones, medias_tiempo, medias_accuracy, ep_optimo, repes, nombre_archivo="ajuste_activacion"):
    x = np.arange(len(activaciones))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tiempo = 'skyblue'
    color_acc = 'salmon'

    ax1.set_xlabel('Función de activación')
    ax1.set_ylabel('Tiempo (s)', color=color_tiempo)
    ax1.bar(x - 0.2, medias_tiempo, width=0.4, color=color_tiempo, label='Tiempo (s)')
    ax1.tick_params(axis='y', labelcolor=color_tiempo)
    ax1.set_xticks(x)
    ax1.set_xticklabels(activaciones)
    ax1.grid(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ax2.bar(x + 0.2, medias_accuracy, width=0.4, color=color_acc, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title(f"Ajuste de función de activación (ép. máx={ep_optimo}, {repes} repeticiones)")

    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre_archivo}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica guardada en {ruta}")
    plt.show()
    plt.close()
def dibuja_NoNeuronas_resultados(activaciones, medias_tiempo, medias_accuracy, ep_optimo, repes, nombre_archivo="ajuste_NoNeuronas"):
    x = np.arange(len(activaciones))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tiempo = 'skyblue'
    color_acc = 'salmon'

    ax1.set_xlabel('No Neuronas')
    ax1.set_ylabel('Tiempo (s)', color=color_tiempo)
    ax1.bar(x - 0.2, medias_tiempo, width=0.4, color=color_tiempo, label='Tiempo (s)')
    ax1.tick_params(axis='y', labelcolor=color_tiempo)
    ax1.set_xticks(x)
    ax1.set_xticklabels(activaciones)
    ax1.grid(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ax2.bar(x + 0.2, medias_accuracy, width=0.4, color=color_acc, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title(f"Ajuste de Numero de Neuronas (ép. máx={ep_optimo}, {repes} repeticiones)")

    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre_archivo}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica guardada en {ruta}")
    plt.show()
    plt.close()
    
    
    
    
def cnn_basica(batch_size = 128, epochs = 15, val_split=0.1):
    # Carga los datos
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10()
    
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"]
        )
    
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=val_split
              )
    # Evaluar
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", score[1])
    print("Test loss:", score[0])

    return model, score[1]

        
def cnn_con_maxpooling(filtro = 3, stride=1, batch_size=128, epochs=10, val_split=0.1):
    # Carga los datos
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10()
    
    # Definir el modelo
    model = Sequential([
        Conv2D(16, (filtro, filtro), strides=stride, activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (filtro, filtro), strides=stride, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy"]
    )
    
    
    # Entrena el modelo
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=val_split,
              verbose=1)
    
    # Evalúa después de entrenar
    score = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test accuracy con filtro {filtro}: {score[1]:.4f}")
    print(f"Test loss con filtro {filtro}: {score[0]:.4f}")
    return model, score[1]


def ajuste_tamano_filtro(filtros=[3,5,7], batch_size=128, epochs=10, val_split=0.1, repes=3):
    tiempos_medios = []
    accuracies_medias = []

    for filtro in filtros:
        tiempos = []
        accuracies = []

        for _ in range(repes):
            start = time.time()
            model, acc = cnn_con_maxpooling(filtro=filtro, batch_size=batch_size, epochs=epochs, val_split=val_split)
            elapsed = time.time() - start

            tiempos.append(elapsed)
            accuracies.append(acc)

        tiempos_medios.append(np.mean(tiempos))
        accuracies_medias.append(np.mean(accuracies))

    dibuja_tamano_filtro_resultados(filtros, tiempos_medios, accuracies_medias, ep_optimo=epochs, repes=repes)
    return filtros, tiempos_medios, accuracies_medias

    
    
def ajuste_stride(strides=[1, 2, 3], batch_size=128, epochs=20, val_split=0.1, repes=5):
    tiempos_medios = []
    accuracies_medias = []

    for stride in strides:
        tiempo_total = 0
        acc_total = 0

        for _ in range(repes):
            start = time.time()
            model, acc_test = cnn_con_maxpooling(stride=stride, batch_size=batch_size, epochs=epochs, val_split=val_split)
            tiempo_total += time.time() - start
            acc_total += acc_test

        tiempos_medios.append(tiempo_total / repes)
        accuracies_medias.append(acc_total / repes)

    # Usa la función que graficaba antes, ajusta parámetros:
    dibuja_stride_resultados(strides, tiempos_medios, accuracies_medias, epochs, repes)
    
    
    
def dibuja_stride_resultados(strides, medias_tiempo, medias_accuracy, ep_optimo, repes, nombre_archivo="ajuste_stride"):
    x = np.arange(len(strides))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tiempo = 'skyblue'
    color_acc = 'salmon'

    ax1.set_xlabel('Stride')
    ax1.set_ylabel('Tiempo (s)', color=color_tiempo)
    ax1.bar(x - 0.2, medias_tiempo, width=0.4, color=color_tiempo, label='Tiempo (s)')
    ax1.tick_params(axis='y', labelcolor=color_tiempo)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strides)
    ax1.grid(axis='y')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ax2.bar(x + 0.2, medias_accuracy, width=0.4, color=color_acc, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title(f"Ajuste de stride (ép. máx={ep_optimo}, {repes} repeticiones)")

    # Guardar y mostrar
    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre_archivo}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica guardada en {ruta}")
    plt.show()
    plt.close()

def dibuja_tamano_filtro_resultados(filtros, medias_tiempo, medias_accuracy, ep_optimo, repes, nombre_archivo="ajuste_tamano_filtro"):
    x = np.arange(len(filtros))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_tiempo = 'skyblue'
    color_acc = 'salmon'

    ax1.set_xlabel('Tamaño filtro (kernel size)')
    ax1.set_ylabel('Tiempo (s)', color=color_tiempo)
    ax1.bar(x - 0.2, medias_tiempo, width=0.4, color=color_tiempo, label='Tiempo (s)')
    ax1.tick_params(axis='y', labelcolor=color_tiempo)
    ax1.set_xticks(x)
    ax1.set_xticklabels(filtros)
    ax1.grid(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    ax2.bar(x + 0.2, medias_accuracy, width=0.4, color=color_acc, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title(f"Ajuste de tamaño de filtro (ép. máx={ep_optimo}, {repes} repeticiones)")

    os.makedirs("graficas", exist_ok=True)
    ruta = f"graficas/{nombre_archivo}.png"
    plt.savefig(ruta)
    print(f"[✔] Gráfica guardada en {ruta}")
    plt.show()
    plt.close()

if __name__ == "__main__":
     
    # Tarea A. Definir, utilizar y evaluar un MLP con Keras
    #MLP((32,32,3), ocultas = 32, activ = "sigmoid", ep = 10, bs = 32, val_split = 0.1, early_stopping=False)
    
    # Tarea B. Ajustar el valor de los parámetros epochs y validation_split
    #ajuste_epochs()
    #ajuste_validation_split(input_shape=(32, 32, 3), ep_optimo=20, repeticiones=5)
    #ajuste_ES(input_shape=(32, 32, 3), ep_max = 50, val_split_optimo = 0.1, repeticiones = 5)
        
    # Tarea C. Ajustar el valor del parámetro batch_size
    #ajuste_batch_size(input_shape=(32, 32, 3), ep_optimo=20, val_split_optimo=0.2, repes=5)
    
    # Tarea D. Probar diferentes funciones de activación
    #ajuste_act_function(input_shape=(32, 32, 3), ep_optimo=20, val_split_optimo=0.1, bs_optimo=64, repeticiones=5)
    
    # Tarea E. Ajustar el número de neuronas por capa
    #ajuste_No_capas(input_shape=(32, 32, 3), ep_optimo=20, val_split_optimo=0.2, bs_optimo=64, repeticiones=5)
    
    # Tarea F. Optimizar un MLP de dos o más capas
    
    
    #Tarea G. CNN Basico
    #cnn_basica(batch_size = 128, epochs = 15, val_split=0.1)
    #cnn_con_maxpooling(batch_size = 128, epochs = 15, val_split=0.1)
    
    # Tarea H. Optimizar el tamaño y el salto de los filtros de convolución.
    #ajuste_tamano_filtro(filtros=[3,5,7], batch_size=128, epochs=10, val_split=0.1, repes=5)
    ajuste_stride(strides=[1,2], batch_size=128, epochs=10, val_split=0.1, repes=5)