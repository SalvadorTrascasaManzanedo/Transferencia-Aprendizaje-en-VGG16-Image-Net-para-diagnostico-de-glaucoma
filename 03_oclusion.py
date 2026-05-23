import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import random
import numpy as np

from tensorflow import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================

model_path = "custom_vgg16_model.keras"
folder = Path("data_set/ACRIMA/Testing/glaucoma")

image_size = (224, 224)
class_names = ["glaucoma", "normal"]
glaucoma_idx = 0

grid_size = 4          # 4x4 parches para el mapa de calor
gray_value = 128
center_square_size = 80

random.seed(42)

# =========================================================
# CARGAR MODELO
# =========================================================

print("1. Cargando modelo...", flush=True)
model = keras.models.load_model(model_path, compile=False)
print("2. Modelo cargado", flush=True)

# =========================================================
# FUNCIONES
# =========================================================

def load_image(path):
    img = keras.utils.load_img(path, target_size=image_size)
    img = keras.utils.img_to_array(img).astype("float32")
    return img

def predict_batch(images):
    batch = np.array(images, dtype="float32")
    return model.predict(batch, verbose=0)

def occlude_center(img, size=80):
    img2 = img.copy()
    h, w, _ = img2.shape

    x1 = w // 2 - size // 2
    x2 = w // 2 + size // 2
    y1 = h // 2 - size // 2
    y2 = h // 2 + size // 2

    img2[y1:y2, x1:x2, :] = gray_value
    return img2

def build_occluded_grid_images(img, grid_size=4):
    h, w, _ = img.shape

    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)

    occluded_images = []
    coords = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = y_edges[row], y_edges[row + 1]
            x1, x2 = x_edges[col], x_edges[col + 1]

            img_occ = img.copy()
            img_occ[y1:y2, x1:x2, :] = gray_value

            occluded_images.append(img_occ)
            coords.append((x1, y1, x2, y2))

    return occluded_images, coords

def compute_occlusion_heatmap(img, pred):
    """
    Calcula un mapa de calor por oclusión para la clase glaucoma.
    """
    original_score = float(pred[glaucoma_idx])

    occluded_images, coords = build_occluded_grid_images(img, grid_size=grid_size)
    preds_occ = predict_batch(occluded_images)

    h, w, _ = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pred_occ, (x1, y1, x2, y2) in zip(preds_occ, coords):
        occ_score = float(pred_occ[glaucoma_idx])
        drop = original_score - occ_score

        # solo nos interesan caídas positivas
        heatmap[y1:y2, x1:x2] = max(drop, 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

# =========================================================
# SELECCIONAR 2 IMÁGENES
# =========================================================

paths = (
    list(folder.glob("*.jpg")) +
    list(folder.glob("*.jpeg")) +
    list(folder.glob("*.png")) +
    list(folder.glob("*.JPG")) +
    list(folder.glob("*.JPEG")) +
    list(folder.glob("*.PNG"))
)

if len(paths) < 2:
    raise ValueError("No hay al menos 2 imágenes en la carpeta glaucoma.")

# si prefieres las 2 primeras, usa: selected_paths = paths[:2]
selected_paths = random.sample(paths, 2)

print("3. Imágenes seleccionadas:", flush=True)
for p in selected_paths:
    print(p.name, flush=True)

# =========================================================
# CARGA DE IMÁGENES
# =========================================================

images = [load_image(p) for p in selected_paths]
images_center_occ = [occlude_center(img, size=center_square_size) for img in images]

# predicción original y post-oclusión central
print("4. Prediciendo originales y tapadas...", flush=True)
preds_original = predict_batch(images)
preds_center_occ = predict_batch(images_center_occ)
print("5. Predicciones terminadas", flush=True)

# =========================================================
# FIGURA FINAL
# =========================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for i, path in enumerate(selected_paths):
    img = images[i]
    img_center = images_center_occ[i]

    pred_orig = preds_original[i]
    pred_post = preds_center_occ[i]

    label_orig = class_names[int(np.argmax(pred_orig))]
    label_post = class_names[int(np.argmax(pred_post))]

    p_g_orig = float(pred_orig[glaucoma_idx])
    p_n_orig = float(pred_orig[1])

    p_g_post = float(pred_post[glaucoma_idx])
    p_n_post = float(pred_post[1])

    print(f"\nImagen: {path.name}", flush=True)
    print(f"Original -> {label_orig} | P(glaucoma)={p_g_orig:.4f} | P(normal)={p_n_orig:.4f}", flush=True)
    print(f"Tapada   -> {label_post} | P(glaucoma)={p_g_post:.4f} | P(normal)={p_n_post:.4f}", flush=True)

    # mapa de calor original
    heatmap_orig = compute_occlusion_heatmap(img, pred_orig)

    # mapa de calor después de tapar el centro
    heatmap_post = compute_occlusion_heatmap(img_center, pred_post)

    # 1) imagen original
    axes[i, 0].imshow(img.astype("uint8"))
    axes[i, 0].set_title(
        f"Original\n{path.name}\nPred: {label_orig}\nP(glaucoma)={p_g_orig:.3f}"
    )
    axes[i, 0].axis("off")

    # 2) imagen original + mapa de calor
    axes[i, 1].imshow(img.astype("uint8"))
    axes[i, 1].imshow(heatmap_orig, cmap="hot", alpha=0.45)
    axes[i, 1].set_title("Mapa de calor\n(original)")
    axes[i, 1].axis("off")

    # 3) imagen con centro tapado
    axes[i, 2].imshow(img_center.astype("uint8"))
    axes[i, 2].set_title(
        f"Centro tapado\nPred: {label_post}\nP(glaucoma)={p_g_post:.3f}"
    )
    axes[i, 2].axis("off")

    # 4) imagen tapada + mapa de calor post
    axes[i, 3].imshow(img_center.astype("uint8"))
    axes[i, 3].imshow(heatmap_post, cmap="hot", alpha=0.45)
    axes[i, 3].set_title("Mapa de calor\n(post-oclusión central)")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.savefig("comparacion_pre_post_occlusion_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n6. Terminado.", flush=True)
print("Figura guardada como: comparacion_pre_post_occlusion_heatmap.png", flush=True)