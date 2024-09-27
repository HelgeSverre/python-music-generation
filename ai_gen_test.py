import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check for GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    print("GPU is available")
else:
    print("No GPU available, using CPU")

# Simple TensorFlow operation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]])
print(tf.matmul(a, b))
