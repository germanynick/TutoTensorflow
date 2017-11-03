import tensorflow as tf

# /cpu:0
# /device:GPU:0
# /device:GPU:1

# Using cpu
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Config
# "log_device_placement", show log
# "allow_soft_placement", automaticalli choose an existing and supported device to run operations
config = tf.ConfigProto(
  log_device_placement=True,
  allow_soft_placement=True,
)

# "allow_growth", allocate as much GPU memory based on runtime allocations.
config.gpu_options.allow_growth = True

# "per_process_gpu_memory_fraction", determines the fraction of the overall
# amount of memory that each visible GPU should be allocated.
config.gpu_options.per_process_gpu_memory_fraction = 0.4


sess = tf.Session(config=config)
print(sess.run(c))
