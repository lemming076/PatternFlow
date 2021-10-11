'''
Test MNIST classifications with JAX and Haiku
'''
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.python.ops.math_ops import reduce_std
import tensorflow_datasets as tfds #stable
#locals
import preprocess

#parameters
epochs = 11 
batch_size = 128
tfdtype = tf.float32
train_set = 'train'
download = True

#===========
#paths and names
home_dir = '/home/shakes/'

#===========
#helper functions
rng = tf.random.Generator.from_seed(123, alg='philox')
# A wrapper function for updating seeds
def f(item):
    seed = rng.make_seeds(2)[0]
    image, label = preprocess.preprocess_samplewise(item, seed)
    return image, label

#===========
#load data
train_ds, info = tfds.load('mnist', split=train_set, shuffle_files=True, data_dir=home_dir+'data', download=download, with_info=True)
total_images = info.splits[train_set].num_examples
total_batches = total_images//batch_size
total_steps = total_batches*epochs
xSize, ySize, rgbSize = info.features['image'].shape
num_classes = info.features['label'].num_classes
get_label_name = info.features['label'].int2str
print("Found", total_images, "training images")
print("No. of Classes:", num_classes)
print("Image Size:", info.features['image'].shape)
print(info)

#training set
train_ds = train_ds.shuffle(10*total_batches)
train_ds = train_ds.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

print("END")
