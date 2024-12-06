import sys
from tqdm import tqdm
from pipelines.gcsstream import get_client, train_datastream
from objects.autoencoder import SparseAutoEncoder
import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# configure autoencoder
name = sys.argv[1]
encoding_size = sys.argv[2]
expansion_factor = sys.argv[3]

try:
    assert(name.isascii() and encoding_size.isnumeric() and expansion_factor.isnumeric())
except AssertionError:
    print("Invalid input. Please provide a valid name, encoding size, and expansion factor.")
    sys.exit(1)
encoding_size = int(encoding_size)
expansion_factor = int(expansion_factor)

model = SparseAutoEncoder(encoding_size=encoding_size, expansion_factor=expansion_factor, name=name)

# cloud data connection
gcs_client = get_client()
bucket = gcs_client.get_bucket("ek990")
dataset = train_datastream(bucket, "sp-embed-tfrecords/*")

# training configuration
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss = keras.losses.MeanSquaredError()
metrics = [
    keras.metrics.MeanSquaredError(),
    keras.metrics.Metric(name='placeholder') # placeholder for training, feature output requires a 2nd metric to appease keras
]

tb_callback = keras.callbacks.TensorBoard(log_dir=f"gs://ek990/autoencoder_logs/{name}")
early_stopping = keras.callbacks.EarlyStopping(monitor="mean_squared_error", min_delta=0.001, patience=20, restore_best_weights=True)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# training
history = model.fit(dataset, epochs=1000, callbacks=[tb_callback, early_stopping])

# save model
model.save(f"gs://ek990/autoencoder_models/{name}.keras")
model.save_weights(f"gs://ek990/autoencoder_models/{name}_weights.h5")
print("Model saved to GCS.")
print(model.summary())

############################################################################################################################################################
######   AUTOMATIC TESTING AND VISUALIZATION OF AUTOENCODER PERFORMANCE   ######
############################################################################################################################################################

print("===== MSE =====")
sns.lineplot(data=history.history['mean_squared_error'], label='Mean Squared Error')
sns.lineplot(history.history['val_mean_squared_error'], label='Validation Mean Squared Error')
plt.title("Mean Squared Error")
plt.show()

print("===== LOSS =====")
sns.lineplot(data=history.history['loss'], label='Loss')
sns.lineplot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.show()

print("===== FEATURE WEIGHT DISTR =====")
feat_weights = np.array(model.weights[1])
print(feat_weights.shape)
agg_feat_weights = np.sum(feat_weights, axis=0)
feat_weights = feat_weights.flatten()
print(feat_weights.max())
print(feat_weights.min())

sns.histplot(feat_weights, kde=True)
plt.title('Distribution of Feature Weights')
plt.xlabel('Feature Weight')
plt.ylabel('Frequency')
plt.show()

sns.histplot(agg_feat_weights, kde=True)
plt.title('Distribution of Per-Neuron Aggregate Feature Weights')
plt.xlabel('Aggregate Feature Weight')
plt.ylabel('Frequency')
plt.show()

print("===== FEATURE OUTPUT DISTR =====")
reconstructed_outputs, feature_outputs = model.predict_on_batch() #TODO

sns.histplot(feature_outputs.flatten(), kde=True)
plt.title('Distribution of feature_outputs')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()

print("===== WEIGHT VECTOR ORTHOGONALITY =====")
def vector_angle(v1, v2):
    """finds the angle between two vectors"""
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def radians_to_degrees(radians):
    return radians * 180 / np.pi

feat_weights = np.array(model.weights[1])

input_vectors = feat_weights.T
output_vectors = feat_weights

del feat_weights
print("Input Vectors Shape:")
print(input_vectors.shape)
print("Output Vectors Shape:")
print(output_vectors.shape)

input_angles = []
output_angles = []

print("Calculating Input Angles...")
for i, v1 in tqdm(enumerate(input_vectors), total=len(input_vectors)):
    for v2 in input_vectors[:i:]:
        input_angles.append(radians_to_degrees(vector_angle(v1, v2)))

print("Calculating Output Angles...")
for i, v1 in tqdm(enumerate(output_vectors), total=len(output_vectors)):
    for v2 in output_vectors[:i:]:
        output_angles.append(radians_to_degrees(vector_angle(v1, v2)))


# Plot histogram of input angles
sns.histplot(input_angles, kde=True)
plt.title('Distribution of Input Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram of output angles
sns.histplot(output_angles, kde=True)
plt.title('Distribution of Output Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.show()