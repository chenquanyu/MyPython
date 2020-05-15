import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def queue_test():
    q = tf.FIFOQueue(3, tf.float32)

    # enqueue
    enq_many = q.enqueue_many([[0.1, 0.2, 0.3], ])

    # add one 
    data = q.dequeue()
    data = data + 1
    enq = q.enqueue(data)

    # run
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run(enq_many))

        for i in range(10):
            sess.run(enq)

        for i in range(3):
            print(sess.run(q.dequeue()))
    return None


if __name__ == "__main__":
    queue_test()
