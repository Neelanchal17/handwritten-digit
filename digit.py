from sklearn.datasets import load_digits
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

digits = load_digits()
pddata = pd.DataFrame(digits.data)

X = pddata.to_numpy()

y = np.array(digits.target)

print(f'X element: {len(X[0])}, Y element: {y[:20]}')
z = 0.8
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    ) # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))


model0 = Sequential([
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='softmax')
])
print(model0.summary())
model0.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
)

model0.fit(
    X_train, y_train,
    epochs=100,
)

preds = model0.predict(X_test)

final_preds = []
print(preds[0])
for k in range(len(preds)):
    final_preds.append(np.argmax(preds[k]))
y_test_new = list(y_test)
print(f'Length of actual output: {len(y_test_new)}, LEngth of predictions: {len(final_preds)}')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test_new, final_preds)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix of Handwritten Digit Classifier")
plt.show()

# Plot some test images and their predicted and actual labels
def plot_misclassified_images(X_test, y_test, final_preds, num_images=10):
    misclassified_indices = np.where(y_test != final_preds)[0]  # Find indices of misclassified examples

    plt.figure(figsize=(12, 12))
    for i, index in enumerate(misclassified_indices[:num_images]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[index].reshape(8, 8), cmap='gray')  # Reshape to original image size (8x8)
        plt.title(f"True: {y_test[index]}, Pred: {final_preds[index]}")
        plt.axis('off')

    plt.suptitle('Misclassified Digits: Actual vs Predicted', fontsize=16)
    plt.show()

# Example usage
plot_misclassified_images(X_test, y_test, final_preds)