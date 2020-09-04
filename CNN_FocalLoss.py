import tensorflow.compat.v1 as tf

import pickle as pk

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

tf.disable_eager_execution()

training_itr = 4000
learning_rate = 1e-3
batch_size = 128

# Different from task to task
num_classes = 3

# Import data
pickle_in_xtrain = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/\
Math/mydata/level/CNN_FocalLoss_Data/xtrain.pickle", "rb")

pickle_in_ytrain = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/\
Math/mydata/level/CNN_FocalLoss_Data/ytrain_onehot.pickle", "rb")

pickle_in_xvalid = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/\
Math/mydata/level/CNN_FocalLoss_Data/xvalid.pickle", "rb")

pickle_in_yvalid = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/\
Math/mydata/level/CNN_FocalLoss_Data/yvalid_onehot.pickle", "rb")

xtrain = pk.load(pickle_in_xtrain)
ytrain = pk.load(pickle_in_ytrain)
xvalid = pk.load(pickle_in_xvalid)
yvalid = pk.load(pickle_in_yvalid)

xvalid = xvalid.reshape((xvalid.shape[0], 128, 768, 1))





# Graph building
x = tf.placeholder(tf.float32, [None, 128, 768, 1], name="x")  # x represents input
y = tf.placeholder(tf.float32, [None, num_classes], name="y")  # y represents output


# Definition of Convolution Layer
def conv2d(x, w, b, stride=1):
    x_1 = tf.nn.conv2d(x, w, strides=[1, stride, 1, 1], padding="VALID")
    x_1 = tf.nn.bias_add(x_1, b)
    return tf.nn.relu(x_1)


# Definition of Pre-maxpooling Layer
def pre_maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 8, 1], strides=[1, 1, 8, 1], padding='VALID')


# Definition of Maxpooling Layer
def maxpool2d(x, k=8):
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')


# Design of weights and biases dictionary
weights = {
    'w_conv_5': tf.get_variable("W5", shape=(5, 96, 1, 2), initializer=tf.truncated_normal_initializer(stddev=0.1)),
    'w_conv_9': tf.get_variable("W9", shape=(9, 96, 1, 2), initializer=tf.truncated_normal_initializer(stddev=0.1)),
    'w_conv_17': tf.get_variable("W17", shape=(17, 96, 1, 2), initializer=tf.truncated_normal_initializer(stddev=0.1)),

    'w_dense1': tf.get_variable("W_dense1", shape=(90, 128), initializer=tf.truncated_normal_initializer(stddev=0.1)),
    'w_dense2': tf.get_variable("W_dense2", shape=(128, 128), initializer=tf.truncated_normal_initializer(stddev=0.1)),
    'w_out': tf.get_variable('W_out', shape=(128, num_classes), initializer=tf.truncated_normal_initializer(stddev=0.1))
}

biases = {
    'b_conv_5': tf.get_variable("B5", shape=(2), initializer=tf.constant_initializer(0.1)),
    'b_conv_9': tf.get_variable("B9", shape=(2), initializer=tf.constant_initializer(0.1)),
    'b_conv_17': tf.get_variable("B17", shape=(2), initializer=tf.constant_initializer(0.1)),

    'b_dense1': tf.get_variable('B_dense1', shape=(128), initializer=tf.constant_initializer(0.1)),
    'b_dense2': tf.get_variable('B_dense2', shape=(128), initializer=tf.constant_initializer(0.1)),
    'b_out': tf.get_variable('B_out', shape=(num_classes), initializer=tf.constant_initializer(0.1))
}

def save_parameters(itr, accuracy):
    for key in weights:
        out = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/"
                   + key + ".pickle", "wb")
        pk.dump(weights[key].eval(), out)
        out.close()

    for key in biases:
        out = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/"
                   + key + ".pickle", "wb")
        pk.dump(biases[key].eval(), out)
        out.close()

    acc = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/max_accuracy.txt", 'w')
    acc.write("Accuracy:   " + str(accuracy) + " at epoch " + str(itr))
    acc.close()

# Definition of Convolution Network
def CNN(x, weights, biases):
    # Convolution part
    x_1 = pre_maxpool(x)
    conv5 = conv2d(x_1, weights['w_conv_5'], biases['b_conv_5'])
    conv9 = conv2d(x_1, weights['w_conv_9'], biases['b_conv_9'])
    conv17 = conv2d(x_1, weights['w_conv_17'], biases['b_conv_17'])

    # Max Pooling
    conv5 = maxpool2d(conv5)
    conv9 = maxpool2d(conv9)
    conv17 = maxpool2d(conv17)

    # Reshape & Concatenation
    conv5 = tf.reshape(tf.transpose(tf.reshape(conv5, [-1, 16, 2]), perm=[0, 2, 1]), [-1, 32])
    conv9 = tf.reshape(tf.transpose(tf.reshape(conv9, [-1, 15, 2]), perm=[0, 2, 1]), [-1, 30])
    conv17 = tf.reshape(tf.transpose(tf.reshape(conv17, [-1, 14, 2]), perm=[0, 2, 1]), [-1, 28])
    dense_input = tf.concat([conv5, conv9, conv17], -1)

    # Fully connected layer
    dense_output_1 = tf.nn.relu(tf.add(tf.matmul(dense_input, weights['w_dense1']), biases['b_dense1']))
    dense_output_2 = tf.nn.relu(tf.add(tf.matmul(dense_output_1, weights['w_dense2']), biases['b_dense2']))
    drop_out = tf.nn.dropout(dense_output_2, rate=0.3)
    output = tf.add(tf.matmul(drop_out, weights['w_out']), biases['b_out'])

    return output


# Definition of Cross_Entropy_Loss(Focal Loss Version)
def focal_cross_entropy_loss(y_pred, y_true):
    epsilon = 1.e-7
    gamma = 2.0

    # different from task to task, number = class number
    alpha = tf.constant([[1], [1.5], [2]], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred)
    log_loss = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    focal_loss = tf.reduce_mean(tf.matmul(tf.multiply(weight, log_loss), alpha))
    return focal_loss


# Prediction, Cost, Optimization
predict = CNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
# cost = tf.reduce_mean(focal_cross_entropy_loss(predict, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation(accuracy)
correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialization
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    """
    summary_writer = tf.summary.FileWriter('C:/Users/Neil Marcus/Desktop/SRP Project\
    /Math/Math/Math/mydata/level/CNN_session', sess.graph)
    """

    acc_max = 0

    for i in range(training_itr):

        if i > 1000:
            learning_rate = 1e-4

        for batch in range(len(xtrain) // batch_size):
            batch_x = xtrain[batch * batch_size: min(len(xtrain), (batch + 1) * batch_size)] \
                .reshape((min(len(xtrain) - batch * batch_size, batch_size), 128, 768, 1))
            batch_y = ytrain[batch * batch_size: min(len(ytrain), (batch + 1) * batch_size)]

            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            loss_train, acc_train = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

        print("Iter " + str(i) + ", Loss= " + \
              "{:.6f}".format(loss_train) + ", Training Accuracy= " + \
              "{:.5f}".format(acc_train))

        print("Optimization Finished!")

        loss_valid, acc_valid = sess.run([cost, accuracy], feed_dict={x: xvalid, y: yvalid})

        if acc_valid > acc_max:
            acc_max = acc_valid
            save_parameters(i, acc_max)

        train_loss.append(loss_train)
        train_accuracy.append(acc_train)
        valid_loss.append(loss_valid)
        valid_accuracy.append(acc_valid)

        print("Testing Accuracy:", "{:.5f}".format(acc_valid))

        plt.ion()
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(valid_accuracy)), valid_accuracy, 'r', label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Acc', fontsize=16)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(valid_loss)), valid_loss, 'b', label='Test Loss')
        plt.title('Test Loss')
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)

        plt.legend()
        plt.pause(1)
        plt.close()
        plt.ioff()

        np.save(
            "C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/valid_acc.npy"
            , valid_accuracy)

        np.save(
            "C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/CNN_FocalLoss_Data/Optimal/valid_loss.npy"
            , valid_loss)

    # summary_writer.close()
