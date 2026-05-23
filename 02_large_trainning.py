import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score,
    fbeta_score
)

import seaborn as sns


# ============================================================
# 0. CONFIGURACIÓN GENERAL
# ============================================================

dataset_path = Path("data_set/ACRIMA")
rimone_path = Path("data_set/RIM-ONE")

# Estructura:
# - data_set/ACRIMA/Training
# - data_set/ACRIMA/dev
# - data_set/ACRIMA/Testing
train_path = dataset_path / "Training"
dev_path = dataset_path / "dev"
test_path = dataset_path / "Testing"

image_size = (224, 224)
batch_size = 32
seed = 42
epochs = 25

output_dir = Path("graficas")
output_dir.mkdir(exist_ok=True)

model_dir = Path("modelos")
model_dir.mkdir(exist_ok=True)

keras.utils.set_random_seed(seed)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

train_ds = keras.utils.image_dataset_from_directory(
    train_path,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Total Classes:", num_classes)

if "glaucoma" not in class_names:
    raise ValueError(f"No se encontró la clase 'glaucoma'. Clases detectadas: {class_names}")

glaucoma_idx = class_names.index("glaucoma")
print("Índice glaucoma:", glaucoma_idx)

val_ds = keras.utils.image_dataset_from_directory(
    dev_path,
    class_names=class_names,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    test_path,
    class_names=class_names,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

rimone_ds = keras.utils.image_dataset_from_directory(
    rimone_path,
    class_names=class_names,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)


def ignore_errors_compat(ds):
    try:
        return ds.ignore_errors()
    except AttributeError:
        return ds.apply(tf.data.experimental.ignore_errors())


# One-hot encode para categorical_crossentropy, como en el tutorial.
train_ds = train_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
    num_parallel_calls=tf.data.AUTOTUNE
)

rimone_ds = rimone_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Protección ligera por si una imagen aislada falla.
train_ds = ignore_errors_compat(train_ds).prefetch(1)
val_ds = ignore_errors_compat(val_ds).prefetch(1)
test_ds = ignore_errors_compat(test_ds).prefetch(1)
rimone_ds = ignore_errors_compat(rimone_ds).prefetch(1)


# ============================================================
# 2. MODELO VGG16 TRANSFER LEARNING
# ============================================================

input_shape = (224, 224, 3)

# Diferencia respecto al tutorial:
# weights="imagenet" porque la práctica pide transferencia de aprendizaje.
# Si pusieras weights=None y congelaras VGG16, la base estaría aleatoria.
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape
)

base_model.trainable = False

inputs = keras.Input(shape=input_shape)

x = preprocess_input(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

# Clasificador superior en la línea del tutorial
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="VGG16_Glaucoma_Transfer"
)

model.summary()


# ============================================================
# 3. COMPILACIÓN
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc")
    ]
)


# ============================================================
# 4. CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(model_dir / "best_vgg16_glaucoma.keras"),
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 5. ENTRENAMIENTO
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# 6. EVALUACIÓN DETALLADA
# ============================================================

def evaluate_dataset(model, ds, dataset_name):
    loss, acc, auc_keras = model.evaluate(ds, verbose=0)

    y_true = []
    y_pred = []
    y_score = []

    for images, labels in ds:
        preds = model.predict(images, verbose=0)

        true_idx = np.argmax(labels.numpy(), axis=1)
        pred_idx = np.argmax(preds, axis=1)

        y_true.extend(true_idx)
        y_pred.extend(pred_idx)
        y_score.extend(preds[:, glaucoma_idx])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    y_true_glaucoma = (y_true == glaucoma_idx).astype(int)

    if len(np.unique(y_true_glaucoma)) == 2:
        auc_manual = roc_auc_score(y_true_glaucoma, y_score)
    else:
        auc_manual = np.nan

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=glaucoma_idx,
        zero_division=0
    )

    f2 = fbeta_score(
        y_true,
        y_pred,
        beta=2,
        pos_label=glaucoma_idx,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    tp = cm[glaucoma_idx, glaucoma_idx]
    fn = cm[glaucoma_idx, :].sum() - tp
    fp = cm[:, glaucoma_idx].sum() - tp
    tn = cm.sum() - tp - fn - fp

    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Keras: {auc_keras:.4f}")
    print(f"AUC manual glaucoma: {auc_manual:.4f}")
    print(f"F1 glaucoma: {f1:.4f}")
    print(f"F2 glaucoma: {f2:.4f}")
    print(f"Sensibilidad glaucoma: {sensibilidad:.4f}")
    print(f"Especificidad normal: {especificidad:.4f}")

    print("\nMatriz de confusión:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    return {
        "name": dataset_name,
        "loss": loss,
        "accuracy": acc,
        "auc_keras": auc_keras,
        "auc": auc_manual,
        "f1": f1,
        "f2": f2,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "y_true_glaucoma": y_true_glaucoma
    }


val_results = evaluate_dataset(model, val_ds, "Dev ACRIMA")
test_results = evaluate_dataset(model, test_ds, "Test ACRIMA")
rimone_results = evaluate_dataset(model, rimone_ds, "Test externo RIM-ONE")


# ============================================================
# 7. GRÁFICAS Y RESUMEN FINAL
# ============================================================

results = {
    "Dev ACRIMA": val_results,
    "Test ACRIMA": test_results,
    "RIM-ONE": rimone_results
}

# ------------------------------------------------------------
# 7.1 Curvas de aprendizaje: accuracy + loss juntas
# ------------------------------------------------------------

epochs_range = range(1, len(history.history["loss"]) + 1)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(epochs_range, history.history["accuracy"], label="Train Accuracy")
ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(epochs_range, history.history["loss"], linestyle="--", label="Train Loss")
ax2.plot(epochs_range, history.history["val_loss"], linestyle=":", label="Validation Loss")
ax2.set_ylabel("Loss")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
plt.title("Learning curves: Accuracy and Loss")

plt.savefig(
    output_dir / "curvas_accuracy_loss.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# 7.2 Curva AUC
# ------------------------------------------------------------

if "auc" in history.history and "val_auc" in history.history:
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history.history["auc"], label="Train AUC")
    plt.plot(epochs_range, history.history["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.title("AUC evolution")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir / "curva_auc.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ------------------------------------------------------------
# 7.3 Matrices de confusión
# ------------------------------------------------------------

def save_confusion_matrix(result, filename):
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        result["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.title(f"Matriz de confusión - {result['name']}")

    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


for key, result in results.items():
    filename = "matriz_confusion_" + key.lower().replace(" ", "_").replace("-", "_") + ".png"
    save_confusion_matrix(result, filename)


# ------------------------------------------------------------
# 7.4 ROC comparativa: Test ACRIMA vs RIM-ONE
# ------------------------------------------------------------

if (
    len(np.unique(test_results["y_true_glaucoma"])) == 2
    and len(np.unique(rimone_results["y_true_glaucoma"])) == 2
):
    fpr_test, tpr_test, _ = roc_curve(
        test_results["y_true_glaucoma"],
        test_results["y_score"]
    )

    fpr_rimone, tpr_rimone, _ = roc_curve(
        rimone_results["y_true_glaucoma"],
        rimone_results["y_score"]
    )

    plt.figure(figsize=(8, 7))

    plt.plot(
        fpr_test,
        tpr_test,
        label=f"ACRIMA Test - AUC = {test_results['auc']:.4f}"
    )

    plt.plot(
        fpr_rimone,
        tpr_rimone,
        label=f"RIM-ONE - AUC = {rimone_results['auc']:.4f}"
    )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Tasa de falsos positivos (1 - especificidad)")
    plt.ylabel("Tasa de verdaderos positivos (sensibilidad)")
    plt.title("Comparación ROC - ACRIMA vs RIM-ONE")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir / "comparacion_roc_acrima_rimone.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ------------------------------------------------------------
# 7.5 Comparación de métricas: Test ACRIMA vs RIM-ONE
# ------------------------------------------------------------

metric_names = ["auc", "f1", "f2", "sensibilidad", "especificidad"]
metric_labels = ["AUC", "F1", "F2", "Sensibilidad", "Especificidad"]

acrima_values = [test_results[m] for m in metric_names]
rimone_values = [rimone_results[m] for m in metric_names]

x = np.arange(len(metric_labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, acrima_values, width, label="ACRIMA Test")
plt.bar(x + width / 2, rimone_values, width, label="RIM-ONE")
plt.xticks(x, metric_labels)
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.title("Comparación de rendimiento - ACRIMA vs RIM-ONE")
plt.legend()
plt.grid(axis="y")

plt.savefig(
    output_dir / "comparacion_metricas_acrima_rimone.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# 7.6 Guardar métricas numéricas en CSV
# ------------------------------------------------------------

csv_path = output_dir / "metricas_finales.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "dataset",
        "loss",
        "accuracy",
        "auc",
        "f1",
        "f2",
        "sensibilidad",
        "especificidad"
    ])

    for name, result in results.items():
        writer.writerow([
            name,
            result.get("loss", np.nan),
            result.get("accuracy", np.nan),
            result["auc"],
            result["f1"],
            result["f2"],
            result["sensibilidad"],
            result["especificidad"]
        ])


# ============================================================
# 8. GUARDAR MODELO
# ============================================================

model.save(model_dir / "custom_vgg16_model.keras")

print("\nModelo guardado correctamente en:")
print(model_dir / "custom_vgg16_model.keras")

print("\nMejor checkpoint guardado en:")
print(model_dir / "best_vgg16_glaucoma.keras")

print("\nGráficas y métricas guardadas en:")
print(output_dir)
print(csv_path)