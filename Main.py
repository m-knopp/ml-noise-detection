import tensorflow as tf
import numpy as np
import data_provider
import tsne
import Graph0
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

def _is_gpu():
    devices = device_lib.list_local_devices()
    #If theres only one device its the cpu:
    if len(devices) == 1:
        return False
    else:
        return True

def get_audio_db_path():
    if _is_gpu():
        return "/home ... ect"
    else:
        return "/home/mknopp/Dropbox/00_BA/Sound_DB"

def tSNE_analysis(dp):
    BATCHSIZE = 50
    batch = dp.get_train_batch(BATCHSIZE)
    imgs, labels = batch[0], batch[1]
    imgs = np.reshape(imgs, [BATCHSIZE, 4096])
    labels = 0.5 + 0.5 * labels
    Y = tsne.tsne(imgs, 2, 20, 20.0)

    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.get_cmap("brg"))
    plt.colorbar()
    plt.show()

def train(data):
    BATCHSIZE = 25

    with tf.Session() as sess:
        graph = tf.get_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, 128, 32], name="inp_imgs")
        y_ = tf.placeholder(tf.float32, shape=[None, 1], name="inp_labels")

        assert x.graph == graph


        #Build Graph/Model
        y = Graph0.Build_Graph(x)

        with tf.name_scope("cost"):
            # Loss function
            loss = tf.losses.mean_squared_error(y, y_)
            # Training Operation
            train_step = tf.train.GradientDescentOptimizer(0.5, name="train").minimize(loss)
            tf.summary.scalar("loss", loss)

        sum_writer = tf.summary.FileWriter("/home/mknopp/tboard_log", graph)
        merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        for n in range(5):                        #EPOCHS
            batch = data.get_train_batch(BATCHSIZE)
            while batch != -1:
                imgs = batch[0]
                noise = np.reshape(batch[1], [BATCHSIZE, 1])
                summary, _ = sess.run([merged, train_step], feed_dict={x: imgs, y_: noise})
                batch = data.get_train_batch(BATCHSIZE)
            print("EPOCH LOSS: " + str(loss.eval(feed_dict={x: imgs, y_: noise})))
            data.load_melspec(0)
            sum_writer.add_summary(summary)

def main():
    #beginns infinite training loop
    dp = data_provider.Data_provider(get_audio_db_path())
    train(dp)
    #tSNE_analysis(dp)


if __name__ == "__main__":
    main()
