import tensorflow as tf
import numpy as np
import cv2
import os

# --- Preparar el Dataset ---

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

#640 x 360
# Tamaño estándar de la imagen
IMG_WIDTH = 640
IMG_HEIGHT = 360


# Directorio del video
video_path = "aa.mp4"

# Directorios de salida
color_dir = "acolor"
gray_dir = "train"

# Crear los directorios si no existen
os.makedirs(color_dir, exist_ok=True)
os.makedirs(gray_dir, exist_ok=True)

# Parámetro para controlar si se generan nuevas imágenes
generate_new_images = False

if generate_new_images:
    # Abrir el vídeo
    cap = cv2.VideoCapture(video_path)

    # Contador de frames
    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

            # Guardar el frame en color
            cv2.imwrite(f"{color_dir}/frame_{frame_count}.jpg", resized_frame)

            # Convertir el frame a escala de grises
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Guardar el frame en escala de grises
            cv2.imwrite(f"{gray_dir}/frame_{frame_count}.jpg", gray)

            frame_count += 1
        else:
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
else:
    # Contar los frames existentes en el directorio
    frame_count = len([name for name in os.listdir(color_dir) if os.path.isfile(os.path.join(color_dir, name))])

# Cargar las imágenes a color y en escala de grises
color_images = []
gray_images = []
for i in range(frame_count):
    color_img = cv2.imread(f"{color_dir}/frame_{i}.jpg")
    gray_img = cv2.imread(f"{gray_dir}/frame_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    color_images.append(color_img / 255.0)  # Normalizar
    gray_images.append(gray_img / 255.0)  # Normalizar

color_images = np.array(color_images)
gray_images = np.array(gray_images)
gray_images = gray_images.reshape(gray_images.shape + (1,))  # Añadir canal para la escala de grises

# --- Construir el Autoencoder ---
# Encoder
input_img = tf.keras.Input(shape=(color_images.shape[1], color_images.shape[2], 3))  # Ajusta el tamaño de la imagen
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)



decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Salida en escala de grises

# Autoencoder
autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# --- Checkpoint para guardar el modelo ---

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Crear el callback de checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=1  # Guardar cada época
)

# --- Verificar si hay un checkpoint guardado ---

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Cargando pesos desde el checkpoint: {latest_checkpoint}")
    autoencoder.load_weights(latest_checkpoint)
else:
    print("No se encontró ningún checkpoint. Entrenando desde cero.")

# --- Entrenar el Autoencoder ---

# Dividir el conjunto de datos en lotes
batch_size = 15  # Ajusta el tamaño del lote según tu capacidad de memoria
num_batches = len(color_images) // batch_size

initial_epoch = int(latest_checkpoint.split('-')[-1].split('.')[0]) if latest_checkpoint else 0
epochs = 5

for epoch in range(initial_epoch, initial_epoch + epochs):  # Ajusta el número de épocas
    epoch_loss = 0
    for batch in range(num_batches):
        batch_color_images = color_images[batch * batch_size:(batch + 1) * batch_size]
        batch_gray_images = gray_images[batch * batch_size:(batch + 1) * batch_size]
        batch_loss = autoencoder.train_on_batch(batch_color_images, batch_gray_images)
        epoch_loss += batch_loss
        print(f"Epoch {epoch + 1}, Batch {batch + 1}, Loss: {batch_loss}")

    epoch_loss /= num_batches
    print(f"Epoch {epoch + 1} completed with average loss: {epoch_loss}")

    # Guardar el modelo al final de cada época
    autoencoder.save_weights(checkpoint_path.format(epoch=epoch + 1))

# --- Guardar el Modelo ---

autoencoder.save("grayscale_autoencoder.h5")