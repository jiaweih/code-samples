import numpy as np
from sklearn import datasets
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


def random_batch(X_train, y_train, batch_size):
    ''' Return random batch of data with batch_size.
    '''
    random_idx = np.random.randint(0, len(X_train), size=batch_size)
    X_batch = X_train[random_idx]
    y_batch = y_train[random_idx].reshape((-1, 1))
    return X_batch, y_batch


n_samples = 10000
data = datasets.make_moons(n_samples=n_samples, noise=0.4)
X_train, X_test, y_train, y_test = \
        train_test_split(data[0], data[1], train_size=0.8, random_state=28)

with tf.name_scope("model"):
    X = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.int32, shape=(None, 1))

    W = tf.Variable(tf.random_normal([2, 1]), name='W')
    b = tf.Variable(tf.zeros([1]), name='b')
    # Function for logistic regression.
    # y_pred = tf.div(1., 1. + tf.exp(-tf.matmul(X, W)+b))
    logits = tf.matmul(X, W) + b
    y_pred = tf.sigmoid(logits)

# Create a log directory that TensorBoard will read from.
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

learning_rate = 0.01
# loss = tf.reduce_mean(-(tf.multiply(y, tf.log(y_pred)) +
#                         tf.multiply(1-y, tf.log(1-y_pred))))
# Loss function for logistic regression.
# log_loss function is equivalent to the above line.
with tf.name_scope("train"):
    loss = tf.losses.log_loss(y, y_pred)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # Create a node in the graph that will evaluate loss.
    loss_summary = tf.summary.scalar('LOSS', loss)
    # Write summaries to logfiles in the log directory.
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epoch = 60
batch_size = 50
n_batch = int(np.ceil(n_samples / batch_size))

with tf.name_scope("init"):
    init = tf.global_variables_initializer()

with tf.name_scope("save"):
    # Create a Saver node.
    saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epoch):
        for batch in range(n_batch):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            _, loss_eval, summary_str = sess.run(
                                            [optimizer, loss, loss_summary],
                                            feed_dict={X: X_batch, y: y_batch})
            if batch == n_batch-1:
                print("Epoch ", epoch, "\tLoss: ", loss_eval)
                # Save the model.
                save_path = saver.save(sess, "/tmp/logistic_model.ckpt")
                file_writer.add_summary(summary_str, epoch)

    # Prediction for test data.
    y_pred_test = y_pred.eval(feed_dict={X: X_test,
                                         y: y_test.reshape((-1, 1))})
    save_path = saver.save(sess, "/tmp/logistic_model_final.ckpt")

file_writer.close()

# Predict 1 if predicted probability is greater than 0.5.
y_pred_test = (y_pred_test >= 0.5)
p_score = precision_score(y_test, y_pred_test)
r_score = recall_score(y_test, y_pred_test)
print("precision_score: ", p_score)
print("recall_score: ", r_score)
