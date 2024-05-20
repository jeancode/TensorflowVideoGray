import tensorflow as tf
import cv2
import numpy as np

# --- Parámetros ---

# Tamaño estándar de la imagen
# 640 x 360

# Tamaño estándar de la imagen
IMG_WIDTH = 640
IMG_HEIGHT = 360

# Ruta al modelo guardado
model_path = "grayscale_autoencoder.h5"

# Ruta al video de prueba
test_video_path = "a.mp4"

# --- Cargar el modelo ---

model = tf.keras.models.load_model(model_path)

# --- Abrir el video de prueba ---

cap = cv2.VideoCapture(test_video_path)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        # --- Preprocesar el frame ---

        # Redimensionar el frame
        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

        # Convertir a float32 y normalizar
        resized_frame = resized_frame.astype(np.float32) / 255.0

        # Agregar una dimensión extra (para el batch)
        resized_frame = np.expand_dims(resized_frame, axis=0)

        # --- Convertir el frame a escala de grises usando el modelo ---

        predicted_gray = model.predict(resized_frame)

        # --- Mostrar el frame original y la predicción ---

        # Convertir la predicción a uint8 y eliminar la dimensión extra
        predicted_gray = (predicted_gray[0] * 255).astype(np.uint8)

        # Mostrar el frame original
        cv2.imshow("Frame Original", frame)

        # Mostrar el frame en escala de grises predicho
        cv2.imshow("Escala de Grises (Predicción)", predicted_gray)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()