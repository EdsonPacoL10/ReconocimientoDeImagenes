import os
from tqdm import tqdm
import random as rn
import numpy as np
import pickle
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def InicioDeProceso():
    def cargar_imagenes_desde_carpeta(carpeta):
        data = []
        for categoria in CATEGORIAS:
            path = os.path.join(carpeta, categoria)
            valor = CATEGORIAS.index(categoria)
            listdir = os.listdir(path)
            for imagen_nombre in tqdm(listdir, desc=categoria):
                try:
                    imagen_ruta = os.path.join(path, imagen_nombre)
                    with open(imagen_ruta, 'rb') as f:
                        imagen = Image.open(f).convert('L')
                        imagen = imagen.resize((IMAGE_SIZE, IMAGE_SIZE))
                        imagen = np.array(imagen)
                    data.append([imagen, valor])
                except Exception as e:
                    pass
        rn.shuffle(data)
        x = []
        y = []
        
        for par in tqdm(data, desc="Procesamiento"):
            x.append(par[0])
            y.append(par[1])
        
        x = np.array(x).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)  # Eliminar la dimensión del canal
        
        with open("x.pickle", "wb") as f:
            pickle.dump(x, f)
        print("Archivo x.pickle creado!")
        
        with open("y.pickle", "wb") as f:
            pickle.dump(y, f)
        print("Archivo y.pickle creado!")

    def entrenar_clasificador():
        with open("x.pickle", "rb") as f:
            x = pickle.load(f)
        
        with open("y.pickle", "rb") as f:
            y = pickle.load(f)
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        
        X_train = X_train.reshape(X_train.shape[0], -1)  # Aplanar los datos de entrenamiento
        X_test = X_test.reshape(X_test.shape[0], -1)  # Aplanar los datos de prueba
        
        clasificador = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        clasificador.fit(X_train, y_train)
        
        score = clasificador.score(X_test, y_test)
        print("Precisión del clasificador:", score)
        
        return clasificador

    def consultar_imagen(clasificador, imagen_ruta):
        imagen = Image.open(imagen_ruta).convert("L")
        imagen = imagen.resize((IMAGE_SIZE, IMAGE_SIZE))
        imagen = np.array(imagen).reshape(1, -1)  # Aplanar la imagen
        
        prediccion = clasificador.predict(imagen)
        clase_prediccion = CATEGORIAS[prediccion[0]]
        
        print("La imagen se clasifica como:", clase_prediccion)
        
        # Mostrar la imagen
        plt.imshow(imagen.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        plt.title("Imagen consultada")
        plt.axis("off")
        plt.show()

    CATEGORIAS = ["Anfibios", "Aracnidos", "Aves", "Insectos","Mamiferos","Peces"]
    IMAGE_SIZE = 100

    if __name__ == "__main__":
        DATADIR = "D:/inteligencia artificial/proyecto/reconocimiento de imagenes"
        
        # Generar y guardar los datos de entrenamiento
        cargar_imagenes_desde_carpeta(DATADIR)
        
        # Entrenar el clasificador
        clasificador = entrenar_clasificador()
        
        # Consultar una imagen
        imagen_ruta = "D:/inteligencia artificial/proyecto/reconocimiento de imagenes/PruebaTres.jfif"  # Ruta de la imagen a consultar
        consultar_imagen(clasificador, imagen_ruta)

