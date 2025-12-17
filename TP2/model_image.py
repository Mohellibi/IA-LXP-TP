
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


BASE_DIR = "./mnist_results"  

MODEL_PATH = os.path.join(BASE_DIR, "mnist_DNN_model.h5")
METRICS_PATH = os.path.join(BASE_DIR, "training_metrics.txt")
ACCURACY_PLOT_PATH = os.path.join(BASE_DIR, "accuracy_plot.png")
LOSS_PLOT_PATH = os.path.join(BASE_DIR, "loss_plot.png")
TRAIN_IMAGES_PATH = os.path.join(BASE_DIR, "train_sample_images.png")
TEST_IMAGES_PATH = os.path.join(BASE_DIR, "test_sample_images.png")
ERRORS_PATH = os.path.join(BASE_DIR, "error_samples.png")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")

os.makedirs(BASE_DIR, exist_ok=True)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test  :", x_test.shape)
print("y_test  :", y_test.shape)


random_indices = random.sample(range(x_train.shape[0]), 5)
for i, idx in enumerate(random_indices):
    img = Image.fromarray(255 - x_train[idx])  # fond blanc
    img_path = os.path.join(BASE_DIR, f"sample_image_{i + 1}.png")
    img.save(img_path)
    print(f"Image sauvegardée : {img_path}")


plt.figure(figsize=(15, 8))
for i in range(10):
    r = random.randint(0, x_train.shape[0] - 1)
    plt.subplot(2, 5, i + 1)
    plt.imshow(255 - x_train[r], cmap="gray")
    plt.title(f"Train [{r}] = {y_train[r]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(TRAIN_IMAGES_PATH)
plt.close()


plt.figure(figsize=(15, 8))
for i in range(10):
    r = random.randint(0, x_test.shape[0] - 1)
    plt.subplot(2, 5, i + 1)
    plt.imshow(255 - x_test[r], cmap="gray")
    plt.title(f"Test [{r}] = {y_test[r]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(TEST_IMAGES_PATH)
plt.close()


x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

with open(METRICS_PATH, "w") as f:
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.savefig(ACCURACY_PLOT_PATH)
plt.savefig(LOSS_PLOT_PATH)
plt.close()


model.save(MODEL_PATH)
print(f"Modèle sauvegardé : {MODEL_PATH}")

predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

errors = [i for i in range(len(x_test)) if y_pred[i] != y_true[i]]
errors = errors[:15]

plt.figure(figsize=(12, 8))
for i, idx in enumerate(errors):
    plt.subplot(3, 5, i + 1)
    plt.imshow(255 - x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"True:{y_true[idx]} Pred:{y_pred[idx]}", color="red")
    plt.axis("off")

plt.tight_layout()
plt.savefig(ERRORS_PATH)
plt.close()


cm = confusion_matrix(y_true, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
disp.plot(cmap="viridis", values_format=".2f")
plt.title("Matrice de confusion")
plt.savefig(CONFUSION_MATRIX_PATH)
plt.show()

print(" Exécution terminée avec succès.")
