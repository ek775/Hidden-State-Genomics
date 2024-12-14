import sys, os
from tqdm import tqdm
from pipelines.gcsstream import get_client, train_datastream, parse_tf_record
from objects.autoencoder import SparseAutoEncoder
import tensorflow as tf
import keras
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# set up logging
TF_CPP_VMODULE=segment=2
convert_graph=2
convert_nodes=2
trt_engine_op=2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(tf.compat.v1.Session().as_default()))

tensorflow_logger = tf.get_logger().setLevel("DEBUG")
#tf.debugging.enable_check_numerics()


############################################################################################################################################################
######   CONFIGURATION   ######
############################################################################################################################################################

# basic params from cmd line and environment
try:
    assert(sys.argv[1].isascii() and sys.argv[2].isnumeric() and sys.argv[3].isnumeric())
except Exception as e:
    print("Invalid input. Please provide a valid name, encoding size, and expansion factor.")
    print("Usage: python3 SAE_train.py <name> <encoding_size> <expansion_factor>")
    sys.exit(e)

tpu: bool = False
try:
    libtpu: int = int(os.environ['TPU_LOAD_LIBRARY'])
    if libtpu == 0:
        tpu = True
except:
    print("TPU_LOAD_LIBRARY not set. Defaulting to CPU/GPU.")
    tpu = False

name: str = str(sys.argv[1])
encoding_size: int = int(sys.argv[2])
expansion_factor: int = int(sys.argv[3])
batch_size: int = 128
epochs: int = 100
checkpoint_file_path: str = f"./autoencoder_models/{name}_partial.weights.h5"

# TPU configuration

# Note that you will need to set the following environment variables:
# TPU_NAME=<current TPU name>
# TPU_LOAD_LIBRARY=0

# You may also need to remove the /tmp directory, which contains the libtpu_lockfile that seems to prevent the TPU server from starting

if tpu == True:
    print("===== Configuring TPU =====")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Connecting to TPU cluster {resolver.cluster_spec().as_dict()['worker']}...")
    tf.config.experimental_connect_to_cluster(resolver)
    print("Initializing TPU system...")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print(f"Tensorflow can access {len(tf.config.list_logical_devices('TPU'))} TPUs.")
    print("===== TPU Ready =====")
    strategy = tf.distribute.TPUStrategy(resolver)



# cloud data connection
gcs_client = get_client()
bucket = gcs_client.get_bucket("ek990")
print("===== Data Connection Established =====")
# get file names from bucket
# note the file names are shuffled within the function
tfrecord_files, num_files = train_datastream(bucket, "sp-embed-tfrecords/*")
# split dataset
print("Partitioning Validation Data...")
# ensure full batches for training
train_size: int = round(num_files * 0.8)
val_size: int = round(num_files * 0.2)

total_steps_train: int = train_size // batch_size
total_steps_val: int = val_size // batch_size

steps_per_epoch: int = total_steps_train // epochs
val_steps: int = total_steps_val // epochs

# list file paths
train, val = tfrecord_files[:train_size], tfrecord_files[train_size:train_size+val_size]
del tfrecord_files # save some memory

def make_dataset(filenames: list[str]) -> tf.data.Dataset:
    return tf.data.TFRecordDataset(
        filenames=filenames, 
        buffer_size=0,
        num_parallel_reads=tf.data.experimental.AUTOTUNE).map(parse_tf_record)

if tpu == True:
    with strategy.scope():
        train = make_dataset(train)
        val = make_dataset(val)
else:
    train = make_dataset(train)
    val = make_dataset(val)

print("--- Data Ready ---")



# training configuration
def compile_model(jit:bool = "auto") -> SparseAutoEncoder:
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss = keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
    metrics = [
        keras.metrics.MeanSquaredError(),
        keras.metrics.Metric(name='placeholder') # placeholder for training, feature output requires a 2nd metric to appease keras
    ]

    model = SparseAutoEncoder(encoding_size=encoding_size, expansion_factor=expansion_factor, name=name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=jit)
    
    # for resuming training
    if os.path.exists(checkpoint_file_path):
        model.load_weights(checkpoint_file_path)
        print("Model weights loaded from GCS.")

    model.call(tf.random.normal((batch_size, encoding_size)))
    tf.print(model.get_compile_config())
    tf.print(model.summary())

    return model


# load model onto appropriate training device
print(f"Configuring Sparse Autoencoder with encoding size {encoding_size} and expansion factor {expansion_factor}...")
if tpu == True:
    with strategy.scope():
        model = compile_model(jit=True)
else:
    model = compile_model()

############################################################################################################################################################
######   TRAINING AUTOENCODER   ######
############################################################################################################################################################

tb_callback = keras.callbacks.TensorBoard(log_dir=f"gs://ek990/autoencoder_logs/{name}")

early_stopping = keras.callbacks.EarlyStopping(
    monitor="mean_squared_error", 
    min_delta=0.001, 
    patience=20, # 20 epochs ~ 20% random sample of embeddings due to optimizations
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_file_path, 
    monitor="mean_squared_error", 
    save_weights_only=True,
    save_best_only=True
)

# train model
history = model.fit(
    x = train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(), 
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_data = val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(),
    validation_steps = val_steps,
    callbacks = [tb_callback, early_stopping, checkpoint],
    verbose = 1
)

# save model
model.save(f"./autoencoder_models/{name}.keras")
model.save_weights(f"./autoencoder_models/{name}.weights.h5")
bucket.blob(f"autoencoder_models/{name}.keras").upload_from_filename(f"./autoencoder_models/{name}.keras")
bucket.blob(f"autoencoder_models/{name}.weights.h5").upload_from_filename(f"./autoencoder_models/{name}.weights.h5")
print("Model saved to GCS.")
print(model.summary())



############################################################################################################################################################
######   AUTOMATIC TESTING AND VISUALIZATION OF AUTOENCODER PERFORMANCE   ######
############################################################################################################################################################

print("Generating Descriptive Statistics...")
# send vis to bucket
vis_path = f"./autoencoder_logs/{name}/Descriptive_Stats/"
os.makedirs(vis_path, exist_ok=True)



print("===== MSE =====")
sns.lineplot(data=history.history['mean_squared_error'], label='Mean Squared Error')
sns.lineplot(history.history['val_mean_squared_error'], label='Validation Mean Squared Error')
plt.title("Mean Squared Error")
plt.savefig(vis_path + "MSE.png")
plt.close()



print("===== LOSS =====")
sns.lineplot(data=history.history['loss'], label='Loss')
sns.lineplot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.savefig(vis_path + "Loss.png")
plt.close()



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
plt.close()

sns.histplot(agg_feat_weights, kde=True)
plt.title('Distribution of Per-Neuron Aggregate Feature Weights')
plt.xlabel('Aggregate Feature Weight')
plt.ylabel('Frequency')
plt.savefig(vis_path + "Aggregate_Feature_Weights.png")
plt.close()



print("===== FEATURE OUTPUT DISTR =====")
# load per-residue heme embeddings
heme_embed_path = "./data/variant_embeddings/all_vars"
heme_embeddings = [np.load(f"{heme_embed_path}/{f}").squeeze() for f in os.listdir(heme_embed_path)]
random.shuffle(heme_embeddings) # randomize since we are looking at only some of the residue embeddings
heme_embeddings = np.vstack(heme_embeddings)

reconstructed_outputs, feature_outputs = model.predict_on_batch(heme_embeddings[1042:2042]) # arbitrary slice of 1000 per residue embeddings

sns.histplot(feature_outputs.flatten(), kde=True)
plt.title('Distribution of feature_outputs')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.savefig(vis_path + "Feature_Outputs.png")
plt.close()



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
plt.close()

# Plot histogram of output angles
sns.histplot(output_angles, kde=True)
plt.title('Distribution of Output Angles')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.savefig(vis_path + "Output_Angles.png")
plt.close()


# export to bucket
print(f"Results saved to {vis_path}")
print("Uploading to Google Cloud Storage...")
for file in os.listdir(vis_path):
    bucket.blob(f"{vis_path}{file}").upload_from_filename(f"{vis_path}{file}")
print("===== DONE =====")