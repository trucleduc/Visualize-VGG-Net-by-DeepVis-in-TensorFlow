import numpy as np
import sys
# path to caffe/python
sys.path.insert(0, "/home/caffe/python")
import caffe
import tensorflow as tf



vgg_model = "vgg-caffe/VGG_ILSVRC_16_layers_deploy.prototxt"
vgg_weights = "vgg-caffe/VGG_ILSVRC_16_layers.caffemodel"
tf_model = "vgg-tensorflow/VGG_ILSVRC_16_layers.ckpt"



np.random.seed(0)
image = 255.0 * np.random.random([224, 224, 3])
caffe_x = image.transpose((2, 0, 1))
caffe_x = np.reshape(caffe_x, [1, caffe_x.shape[0], caffe_x.shape[1], caffe_x.shape[2]])
tf_x = np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])



caffe_net = caffe.Net(vgg_model, vgg_weights, caffe.TEST)
caffe_W_conv1_1 = caffe_net.params["conv1_1"][0].data
caffe_b_conv1_1 = caffe_net.params["conv1_1"][1].data
caffe_W_conv1_2 = caffe_net.params["conv1_2"][0].data
caffe_b_conv1_2 = caffe_net.params["conv1_2"][1].data
caffe_W_conv2_1 = caffe_net.params["conv2_1"][0].data
caffe_b_conv2_1 = caffe_net.params["conv2_1"][1].data
caffe_W_conv2_2 = caffe_net.params["conv2_2"][0].data
caffe_b_conv2_2 = caffe_net.params["conv2_2"][1].data
caffe_W_conv3_1 = caffe_net.params["conv3_1"][0].data
caffe_b_conv3_1 = caffe_net.params["conv3_1"][1].data
caffe_W_conv3_2 = caffe_net.params["conv3_2"][0].data
caffe_b_conv3_2 = caffe_net.params["conv3_2"][1].data
caffe_W_conv3_3 = caffe_net.params["conv3_3"][0].data
caffe_b_conv3_3 = caffe_net.params["conv3_3"][1].data
caffe_W_conv4_1 = caffe_net.params["conv4_1"][0].data
caffe_b_conv4_1 = caffe_net.params["conv4_1"][1].data
caffe_W_conv4_2 = caffe_net.params["conv4_2"][0].data
caffe_b_conv4_2 = caffe_net.params["conv4_2"][1].data
caffe_W_conv4_3 = caffe_net.params["conv4_3"][0].data
caffe_b_conv4_3 = caffe_net.params["conv4_3"][1].data
caffe_W_conv5_1 = caffe_net.params["conv5_1"][0].data
caffe_b_conv5_1 = caffe_net.params["conv5_1"][1].data
caffe_W_conv5_2 = caffe_net.params["conv5_2"][0].data
caffe_b_conv5_2 = caffe_net.params["conv5_2"][1].data
caffe_W_conv5_3 = caffe_net.params["conv5_3"][0].data
caffe_b_conv5_3 = caffe_net.params["conv5_3"][1].data
caffe_W_fc6 = caffe_net.params["fc6"][0].data
caffe_b_fc6 = caffe_net.params["fc6"][1].data
caffe_W_fc7 = caffe_net.params["fc7"][0].data
caffe_b_fc7 = caffe_net.params["fc7"][1].data
caffe_W_fc8 = caffe_net.params["fc8"][0].data
caffe_b_fc8 = caffe_net.params["fc8"][1].data



caffe_net.blobs["data"].reshape(*caffe_x.shape)
caffe_net.blobs["data"].data[...] = caffe_x
caffe_net.forward()



x = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
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
sess.run(W_conv1_1.assign(caffe_W_conv1_1.transpose((2, 3, 1, 0))))
sess.run(b_conv1_1.assign(caffe_b_conv1_1))
sess.run(W_conv1_2.assign(caffe_W_conv1_2.transpose((2, 3, 1, 0))))
sess.run(b_conv1_2.assign(caffe_b_conv1_2))
sess.run(W_conv2_1.assign(caffe_W_conv2_1.transpose((2, 3, 1, 0))))
sess.run(b_conv2_1.assign(caffe_b_conv2_1))
sess.run(W_conv2_2.assign(caffe_W_conv2_2.transpose((2, 3, 1, 0))))
sess.run(b_conv2_2.assign(caffe_b_conv2_2))
sess.run(W_conv3_1.assign(caffe_W_conv3_1.transpose((2, 3, 1, 0))))
sess.run(b_conv3_1.assign(caffe_b_conv3_1))
sess.run(W_conv3_2.assign(caffe_W_conv3_2.transpose((2, 3, 1, 0))))
sess.run(b_conv3_2.assign(caffe_b_conv3_2))
sess.run(W_conv3_3.assign(caffe_W_conv3_3.transpose((2, 3, 1, 0))))
sess.run(b_conv3_3.assign(caffe_b_conv3_3))
sess.run(W_conv4_1.assign(caffe_W_conv4_1.transpose((2, 3, 1, 0))))
sess.run(b_conv4_1.assign(caffe_b_conv4_1))
sess.run(W_conv4_2.assign(caffe_W_conv4_2.transpose((2, 3, 1, 0))))
sess.run(b_conv4_2.assign(caffe_b_conv4_2))
sess.run(W_conv4_3.assign(caffe_W_conv4_3.transpose((2, 3, 1, 0))))
sess.run(b_conv4_3.assign(caffe_b_conv4_3))
sess.run(W_conv5_1.assign(caffe_W_conv5_1.transpose((2, 3, 1, 0))))
sess.run(b_conv5_1.assign(caffe_b_conv5_1))
sess.run(W_conv5_2.assign(caffe_W_conv5_2.transpose((2, 3, 1, 0))))
sess.run(b_conv5_2.assign(caffe_b_conv5_2))
sess.run(W_conv5_3.assign(caffe_W_conv5_3.transpose((2, 3, 1, 0))))
sess.run(b_conv5_3.assign(caffe_b_conv5_3))
caffe_W_fc6 = np.reshape(caffe_W_fc6, [4096, 512, 7, 7])
caffe_W_fc6 = caffe_W_fc6.transpose((0, 2, 3, 1))
caffe_W_fc6 = np.reshape(caffe_W_fc6, [4096, 7 * 7 * 512])
sess.run(W_fc6.assign(caffe_W_fc6.transpose(1, 0)))
sess.run(b_fc6.assign(caffe_b_fc6))
sess.run(W_fc7.assign(caffe_W_fc7.transpose(1, 0)))
sess.run(b_fc7.assign(caffe_b_fc7))
sess.run(W_fc8.assign(caffe_W_fc8.transpose(1, 0)))
sess.run(b_fc8.assign(caffe_b_fc8))
saver.save(sess, tf_model)
print("Model saved in file: %s" % tf_model)
tf_hconv = sess.run(h_prob, feed_dict = {x: tf_x, keep_prob: 1.0})
sess.close()



