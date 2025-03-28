import os
import tensorflow as tf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load preprocessed datasets
from dataprocessing import train_dataset, test_dataset

# Define class names explicitly
class_names = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']  # Replace with actual class names

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Save the model
model.save("clothing_model.h5")
print("Model saved as clothing_model.h5")

# Evaluate the model
test_images, test_labels = zip(*list(test_dataset.unbatch().as_numpy_iterator()))
test_images = tf.convert_to_tensor(test_images)
test_labels = tf.convert_to_tensor(test_labels)

predictions = model.predict(test_images)
predicted_labels = tf.argmax(predictions, axis=1)

# Function to plot evaluation metrics
def plot_metrics(metrics, metric_names, title):
    plt.bar(metric_names, metrics, color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(title)
    plt.ylim(0, 1)  # Metrics are typically between 0 and 1
    plt.show()

# Calculate evaluation metrics
print("Classification Report:")
report = classification_report(test_labels, predicted_labels, target_names=class_names, output_dict=True)
print(classification_report(test_labels, predicted_labels, target_names=class_names))

# Extract metrics for plotting
precision = [report[class_name]['precision'] for class_name in class_names]
recall = [report[class_name]['recall'] for class_name in class_names]
f1_score = [report[class_name]['f1-score'] for class_name in class_names]

# Plot metrics
plot_metrics(precision, class_names, "Precision per Class")
plot_metrics(recall, class_names, "Recall per Class")
plot_metrics(f1_score, class_names, "F1-Score per Class")
