import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(6.0)
sum = tf.add(a,b)

graph = tf.get_default_graph()
print(graph)

# placeholder
plt = tf.placeholder(tf.float32, [None, 3])
print(plt)

# variable
va = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0))

init_op = tf.global_variables_initializer()

with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init_op)
    # write graph
    filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

    print(sess.run(sum))
    print(sess.run(plt, feed_dict ={plt:[[1,2,3],[4,5,6]]}))
    #print(plt.op)

plt.set_shape([2,3])
print(plt)

plt2 = tf.reshape(plt, [3,2])
print(plt2)

