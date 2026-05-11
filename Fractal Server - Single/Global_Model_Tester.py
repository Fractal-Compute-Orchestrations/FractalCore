import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# === 1. Rebuild model ===
def build_model():
    # Same architecture as used during training
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # adjust if needed
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model = build_model()

# === 2. Restore from checkpoint ===
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore('./global_model/global.ckpt').expect_partial()  # or use .assert_existing_objects_matched()

# === 3. Run inference ===
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Assuming test_images and test_labels are already loaded
# Shape: test_images.shape = (N, 28, 28)
# Normalize if needed: test_images = test_images / 255.0

pred_probs = model.predict(test_images)
predictions = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(test_labels, axis=1)  # If test_labels is one-hot

# === 4. Plot results ===
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot(images, predictions, true_labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        color = 'b' if predictions[i] == true_labels[i] else 'r'
        plt.xlabel(class_names[predictions[i]], color=color)
    plt.show()

plot(test_images, predictions, true_labels)

print(predictions.shape)
