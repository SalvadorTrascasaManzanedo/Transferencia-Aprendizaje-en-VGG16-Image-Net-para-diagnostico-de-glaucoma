import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --------- Paso 1: Carga del conjunto ACRIMA de prueba ---------

dataset_path = "data_set/ACRIMA/test"

image_size = (224, 224)
batch_size = 32

test_ds = keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("Clases:", class_names)
print("Número de clases:", num_classes)

# Codificación one-hot y preprocesado específico de VGG16
test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), tf.one_hot(y, depth=num_classes))
)

# --------- Paso 2: Construcción de VGG16 con pesos de ImageNet, sin entrenamiento ---------

input_shape = (224, 224, 3)

# Base convolucional preentrenada en ImageNet
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape
)

# Se congelan todas sus capas
base_model.trainable = False

# Entrada del modelo
inputs = keras.Input(shape=input_shape)

# Paso por la base VGG16
x = base_model(inputs)

# Cabeza binaria nueva, pero sin entrenar
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

# Modelo final
model = keras.Model(inputs, outputs, name="VGG16_ImageNet_sin_entrenamiento")

model.summary()

# --------- Paso 3: Predicción sobre ACRIMA/test ---------

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# --------- Paso 4: Evaluación ---------

print("\nInforme de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
plt.title("Matriz de confusión: VGG16 con pesos de ImageNet sin entrenamiento")
plt.show()
