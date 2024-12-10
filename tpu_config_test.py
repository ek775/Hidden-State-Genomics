import tensorflow as tf
print("Tensorflow version " + tf.__version__)

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
