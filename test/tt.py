import tensorflow as tf

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))