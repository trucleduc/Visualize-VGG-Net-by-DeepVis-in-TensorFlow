import numpy as np
import scipy.misc
import scipy.ndimage
import tensorflow as tf



pretrained_model = "vgg-tensorflow/VGG_ILSVRC_16_layers.ckpt"
base_lr = 1000.0
lr_step = 1000
gamma = 0.5
decay = 1e-4
blur_radius = 1.0
blur_every = 4
max_iters = 1000
test_iters = 100
vgg_mean = [103.939, 116.779, 123.68]



x = tf.placeholder(tf.float32, shape = [1, 224, 224, 3])
keep_prob = tf.placeholder(tf.float32)

W_conv1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev = 0.01), name = "W_conv1_1")
b_conv1_1 = tf.Variable(tf.truncated_normal([64], stddev = 0.01), name = "b_conv1_1")
W_conv1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01), name = "W_conv1_2")
b_conv1_2 = tf.Variable(tf.truncated_normal([64], stddev = 0.01), name = "b_conv1_2")
W_conv2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev = 0.01), name = "W_conv2_1")
b_conv2_1 = tf.Variable(tf.truncated_normal([128], stddev = 0.01), name = "b_conv2_1")
W_conv2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev = 0.01), name = "W_conv2_2")
b_conv2_2 = tf.Variable(tf.truncated_normal([128], stddev = 0.01), name = "b_conv2_2")
W_conv3_1 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev = 0.01), name = "W_conv3_1")
b_conv3_1 = tf.Variable(tf.truncated_normal([256], stddev = 0.01), name = "b_conv3_1")
W_conv3_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev = 0.01), name = "W_conv3_2")
b_conv3_2 = tf.Variable(tf.truncated_normal([256], stddev = 0.01), name = "b_conv3_2")
W_conv3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev = 0.01), name = "W_conv3_3")
b_conv3_3 = tf.Variable(tf.truncated_normal([256], stddev = 0.01), name = "b_conv3_3")
W_conv4_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev = 0.01), name = "W_conv4_1")
b_conv4_1 = tf.Variable(tf.truncated_normal([512], stddev = 0.01), name = "b_conv4_1")
W_conv4_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = "W_conv4_2")
b_conv4_2 = tf.Variable(tf.truncated_normal([512], stddev = 0.01), name = "b_conv4_2")
W_conv4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = "W_conv4_3")
b_conv4_3 = tf.Variable(tf.truncated_normal([512], stddev = 0.01), name = "b_conv4_3")
W_conv5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = "W_conv5_1")
b_conv5_1 = tf.Variable(tf.truncated_normal([512], stddev = 0.01), name = "b_conv5_1")
W_conv5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = "W_conv5_2")
b_conv5_2 = tf.Variable(tf.truncated_normal([512], stddev = 0.01), name = "b_conv5_2")
W_conv5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev = 0.01), name = "W_conv5_3")
b_conv5_3 = tf.Variable(tf.truncated_normal([512], stddev = 0.01), name = "b_conv5_3")
W_fc6 = tf.Variable(tf.truncated_normal([7 * 7 * 512, 4096], stddev = 0.01), name = "W_fc6")
b_fc6 = tf.Variable(tf.truncated_normal([4096], stddev = 0.01), name = "b_fc6")
W_fc7 = tf.Variable(tf.truncated_normal([4096, 4096], stddev = 0.01), name = "W_fc7")
b_fc7 = tf.Variable(tf.truncated_normal([4096], stddev = 0.01), name = "b_fc7")
W_fc8 = tf.Variable(tf.truncated_normal([4096, 1000], stddev = 0.01), name = "W_fc8")
b_fc8 = tf.Variable(tf.truncated_normal([1000], stddev = 0.01), name = "b_fc8")

h_conv1_1 = tf.nn.relu(tf.add(tf.nn.conv2d(x, W_conv1_1, strides = [1, 1, 1, 1], padding = "SAME"), b_conv1_1))
h_conv1_2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv1_1, W_conv1_2, strides = [1, 1, 1, 1], padding = "SAME"), b_conv1_2))
h_pool1 = tf.nn.max_pool(h_conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
h_conv2_1 = tf.nn.relu(tf.add(tf.nn.conv2d(h_pool1, W_conv2_1, strides = [1, 1, 1, 1], padding = "SAME"), b_conv2_1))
h_conv2_2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv2_1, W_conv2_2, strides = [1, 1, 1, 1], padding = "SAME"), b_conv2_2))
h_pool2 = tf.nn.max_pool(h_conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
h_conv3_1 = tf.nn.relu(tf.add(tf.nn.conv2d(h_pool2, W_conv3_1, strides = [1, 1, 1, 1], padding = "SAME"), b_conv3_1))
h_conv3_2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv3_1, W_conv3_2, strides = [1, 1, 1, 1], padding = "SAME"), b_conv3_2))
h_conv3_3 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv3_2, W_conv3_3, strides = [1, 1, 1, 1], padding = "SAME"), b_conv3_3))
h_pool3 = tf.nn.max_pool(h_conv3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
h_conv4_1 = tf.nn.relu(tf.add(tf.nn.conv2d(h_pool3, W_conv4_1, strides = [1, 1, 1, 1], padding = "SAME"), b_conv4_1))
h_conv4_2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv4_1, W_conv4_2, strides = [1, 1, 1, 1], padding = "SAME"), b_conv4_2))
h_conv4_3 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv4_2, W_conv4_3, strides = [1, 1, 1, 1], padding = "SAME"), b_conv4_3))
h_pool4 = tf.nn.max_pool(h_conv4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
h_conv5_1 = tf.nn.relu(tf.add(tf.nn.conv2d(h_pool4, W_conv5_1, strides = [1, 1, 1, 1], padding = "SAME"), b_conv5_1))
h_conv5_2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv5_1, W_conv5_2, strides = [1, 1, 1, 1], padding = "SAME"), b_conv5_2))
h_conv5_3 = tf.nn.relu(tf.add(tf.nn.conv2d(h_conv5_2, W_conv5_3, strides = [1, 1, 1, 1], padding = "SAME"), b_conv5_3))
h_pool5 = tf.nn.max_pool(h_conv5_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
h_pool5_flat = tf.reshape(h_pool5, [-1, 7 * 7 * 512])
h_fc6 = tf.nn.relu(tf.add(tf.matmul(h_pool5_flat, W_fc6), b_fc6))
h_fc6_dropout = tf.nn.dropout(h_fc6, keep_prob)
h_fc7 = tf.nn.relu(tf.add(tf.matmul(h_fc6_dropout, W_fc7), b_fc7))
h_fc7_dropout = tf.nn.dropout(h_fc7, keep_prob)
h_fc8 = tf.add(tf.matmul(h_fc7_dropout, W_fc8), b_fc8)
h_prob = tf.nn.softmax(h_fc8)



saver = tf.train.Saver({"W_conv1_1": W_conv1_1, "b_conv1_1": b_conv1_1, "W_conv1_2": W_conv1_2, "b_conv1_2": b_conv1_2, "W_conv2_1": W_conv2_1, "b_conv2_1": b_conv2_1, "W_conv2_2": W_conv2_2, "b_conv2_2": b_conv2_2, "W_conv3_1": W_conv3_1, "b_conv3_1": b_conv3_1, "W_conv3_2": W_conv3_2, "b_conv3_2": b_conv3_2, "W_conv3_3": W_conv3_3, "b_conv3_3": b_conv3_3, "W_conv4_1": W_conv4_1, "b_conv4_1": b_conv4_1, "W_conv4_2": W_conv4_2, "b_conv4_2": b_conv4_2, "W_conv4_3": W_conv4_3, "b_conv4_3": b_conv4_3, "W_conv5_1": W_conv5_1, "b_conv5_1": b_conv5_1, "W_conv5_2": W_conv5_2, "b_conv5_2": b_conv5_2, "W_conv5_3": W_conv5_3, "b_conv5_3": b_conv5_3, "W_fc6": W_fc6, "b_fc6": b_fc6, "W_fc7": W_fc7, "b_fc7": b_fc7, "W_fc8": W_fc8, "b_fc8": b_fc8})
sess = tf.InteractiveSession()
saver.restore(sess, pretrained_model)



fo = open("vgg-caffe/log.txt", "w")
fo.truncate()
for neuron_idx in range(1000):
    activation = h_prob[0, neuron_idx]
    xgrad = tf.gradients(activation, [x])[0]

    x_val = np.float32(np.random.normal(loc = 0.0, scale = 10.0, size = (1, 224, 224, 3)))
    max_x_val, max_activation_val = x_val, 0.0
    lr_val = base_lr
    for iters in range(max_iters):
        if (iters > 0 and iters % lr_step == 0):
            learning_rate *= gamma
        xgrad_val = sess.run(xgrad, feed_dict = {x: x_val, keep_prob: 1.0})
        activation_val = sess.run(activation, feed_dict = {x: x_val, keep_prob: 1.0})
        if (activation_val > max_activation_val):
            max_x_val, max_activation_val = x_val, activation_val
        if (max_activation_val > 0.99):
            break
        if (activation_val > 0.9):
            lr_val = base_lr / 10.0

        print("Class %d, iters %d, learning rate = %f" % (neuron_idx, iters, lr_val))
        print("Activation: ", activation_val)
        print("X: ", np.min(np.ravel(x_val)), np.max(np.ravel(x_val)), np.sqrt(np.sum(np.ravel(x_val) ** 2)))
        print("Grad: ", np.min(np.ravel(xgrad_val)), np.max(np.ravel(xgrad_val)), np.sqrt(np.sum(np.ravel(xgrad_val) ** 2)))
        print("\n")

        # weight decay
        x_val = (1.0 - decay) * (x_val + lr_val * xgrad_val)
        # gaussian blur
        if (iters % blur_every == 0 and iters != max_iters - 1):
            for channel in range(3):
                x_val[0, :, :, channel] = scipy.ndimage.filters.gaussian_filter(x_val[0, :, :, channel], blur_radius)

        if (iters % test_iters == 0 and iters != max_iters - 1):
            I = np.ravel(x_val[0, :, :, :])
            n = np.int32(np.sqrt(np.int32(I.shape[0] / 3)))
            I = np.reshape(I, [n, n, 3])
            scipy.misc.imsave("per-class-images/prob-" + str(neuron_idx).zfill(3) + ".png", I)

    fo.write(str(neuron_idx))
    fo.write(" ")
    fo.write(str(max_activation_val))
    fo.write("\n")
    I = np.ravel(max_x_val[0, :, :, :])
    n = np.int32(np.sqrt(np.int32(I.shape[0] / 3)))
    I = np.reshape(I, [n, n, 3])
    scipy.misc.imsave("per-class-images/prob-" + str(neuron_idx).zfill(3) + ".png", I)

fo.close()
sess.close()

