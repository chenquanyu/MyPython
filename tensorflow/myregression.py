import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def  myregression():
    """ 使用梯度递减实现线性回归  """

    with tf.variable_scope("data"):
        # 1.准备数据
        x = tf.random_normal([100,1], mean=1.75, stddev=0.5, name="x_data")
        # assuming y = 0.7 * x + 0.8
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2. 建立线性回归模型
        weight = tf.Variable(tf.random_normal([1,1],mean=0.0, stddev= 1.0), name = "w")
        bias =  tf.Variable(tf.random_normal([1,1],mean=0.0, stddev= 1.0), name = "b")

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 3. 建立损失函数
        loss = tf.reduce_mean(tf.square(y_true-y_predict))

    with tf.variable_scope("train"):
        # 4. 梯度下降优化损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # collect tensor
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("weight", weight)

    merged = tf.summary.merge_all()
     
    init_op = tf.global_variables_initializer()
    summaryPath = "./tmp/summary/"
    savePath = "./tmp/checkpoints/"

    # init saver
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        sess.run(init_op)
        print("init w: %f, b: %f"%(weight.eval(), bias.eval()))

        # 建立事件文件
        fileWriter = tf.summary.FileWriter(summaryPath, graph=sess.graph)

        # restore parameters
        if os.path.exists(savePath) :
            saver.restore(sess, savePath)

        for i in range(10):
            sess.run(train_op)

            summary = sess.run(merged)
            fileWriter.add_summary(summary, i)

            print("%i times, w: %f, b: %f"%(i, weight.eval(), bias.eval()))

        saver.save(sess, savePath)

    return None

if __name__ == "__main__":
    myregression()