import sys, os
from tqdm import tqdm
from pipelines.gcsstream import get_client, train_datastream, parse_tf_record
from objects.autoencoder import SparseAutoEncoder
import tensorflow as tf
import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


############################################################################################################################################################
######   CONFIGURATION   ######
############################################################################################################################################################

try:
    assert(sys.argv[1].isascii() and sys.argv[2].isnumeric() and sys.argv[3].isnumeric())
except Exception as e:
    print("Invalid input. Please provide a valid name, encoding size, and expansion factor.")
    print("Usage: python3 SAE_train.py <name> <encoding_size> <expansion_factor>")
    sys.exit(e)

name: str = str(sys.argv[1])
encoding_size: int = int(sys.argv[2])
expansion_factor: int = int(sys.argv[3])
batch_size: int = 1000

# TPU configuration
print("===== Testing TPU Configuration =====")

# test TPU cluster connection
os.system("python3 tpu_config_test.py")

print("===== TPU Connection Test Complete =====")


# configure TPU
print("Configuring TPU for training...")
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
print(f"Connecting to TPU cluster {resolver.cluster_spec().as_dict()['worker']}...")
tf.config.experimental_connect_to_cluster(resolver)
print("Initializing TPU system...")
tf.tpu.experimental.initialize_tpu_system(resolver)
print(f"Tensorflow can access {len(tf.config.list_logical_devices('TPU'))} TPUs.")
print("===== TPU Ready =====")
strategy = tf.distribute.experimental.TPUStrategy(resolver)


# cloud data connection
gcs_client = get_client()
bucket = gcs_client.get_bucket("ek990")
print("===== Data Connection Established =====")
dataset = train_datastream(bucket, "sp-embed-tfrecords/*")  # note that shuffling is done within this function for performance reasons
# split dataset
print("Partitioning Validation Data...")
val = dataset.take(int(1e5))
val = val.map(parse_tf_record)
train = dataset.skip(int(1e5)).take(int(5e5))
train = train.map(parse_tf_record)
print("--- Data Ready ---")


# training configuration
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss = keras.losses.MeanSquaredError(reduction="sum")
metrics = [
    keras.metrics.MeanSquaredError(),
    keras.metrics.Metric(name='placeholder') # placeholder for training, feature output requires a 2nd metric to appease keras
]

tb_callback = keras.callbacks.TensorBoard(log_dir=f"gs://ek990/autoencoder_logs/{name}")
early_stopping = keras.callbacks.EarlyStopping(monitor="mean_squared_error", min_delta=0.001, patience=20, restore_best_weights=True)

print(f"Configuring Sparse Autoencoder with encoding size {encoding_size} and expansion factor {expansion_factor}...")
with strategy.scope():
    model = SparseAutoEncoder(encoding_size=encoding_size, expansion_factor=expansion_factor, name=name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

############################################################################################################################################################
######   TRAINING AUTOENCODER   ######
############################################################################################################################################################

history = model.fit(
    x = train.batch(batch_size), 
    epochs = 1000,
    validation_data = val.batch(batch_size),
    callbacks = [tb_callback, early_stopping],
    verbose = 2
    )

# save model
model.save(f"gs://ek990/autoencoder_models/{name}.keras")
model.save_weights(f"gs://ek990/autoencoder_models/{name}_weights.h5")
print("Model saved to GCS.")
print(model.summary())



############################################################################################################################################################
######   AUTOMATIC TESTING AND VISUALIZATION OF AUTOENCODER PERFORMANCE   ######
############################################################################################################################################################

print("Generating Descriptive Statistics...")
# send vis to bucket
vis_path = f"gs://ek990/autoencoder_logs/{name}/Descriptive_Stats/"

print("===== MSE =====")
sns.lineplot(data=history.history['mean_squared_error'], label='Mean Squared Error')
sns.lineplot(history.history['val_mean_squared_error'], label='Validation Mean Squared Error')
plt.title("Mean Squared Error")
plt.savefig(vis_path + "MSE.png")

print("===== LOSS =====")
sns.lineplot(data=history.history['loss'], label='Loss')
sns.lineplot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.savefig(vis_path + "Loss.png")

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
plt.savefig(vis_path + "Feature_Weights.png")

sns.histplot(agg_feat_weights, kde=True)
plt.title('Distribution of Per-Neuron Aggregate Feature Weights')
plt.xlabel('Aggregate Feature Weight')
plt.ylabel('Frequency')
plt.savefig(vis_path + "Aggregate_Feature_Weights.png")

print("===== FEATURE OUTPUT DISTR =====")
# load per-residue heme embeddings
heme_embed_path = "./data/variant_embeddings/all_vars"
heme_embeddings = [np.load(f"{heme_embed_path}/{f}").squeeze() for f in os.listdir(heme_embed_path)]
heme_embeddings = np.vstack(heme_embeddings)

reconstructed_outputs, feature_outputs = model.predict_on_batch(heme_embeddings[1042:2042]) # arbitrary slice of 1000 per residue embeddings

sns.histplot(feature_outputs.flatten(), kde=True)
plt.title('Distribution of feature_outputs')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.savefig(vis_path + "Feature_Outputs.png")

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
plt.savefig(vis_path + "Input_Angles.png")

# Plot histogram of output angles
sns.histplot(output_angles, kde=True)
plt.title('Distribution of Output Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.savefig(vis_path + "Output_Angles.png")