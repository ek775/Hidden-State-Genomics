import tensorflow as tf
import os
print("Tensorflow version " + tf.__version__)

os.system("export TPU_NAME=tpu-vm-1")
os.system("export TPU_LOAD_LIBRARY=0")
os.system("sudo rm -r /tmp/") # removing lockfile that seems to prevent tpu server startup

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
print('Running on TPU ', cluster_resolver.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

@tf.function
def add_fn(x,y):
  z = x + y
  return z

x = tf.constant(1.)
y = tf.constant(1.)
z = strategy.run(add_fn, args=(x,y))
print(z)
