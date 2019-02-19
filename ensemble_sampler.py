import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

def xavier_init(n_inputs, n_outputs, uniform=True):
  if uniform:
    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def discriminator(x):
	with tf.variable_scope("D"):
	    D_W1 = tf.Variable(xavier_init([784, 128]))
        D_b1 = tf.Variable(tf.zeros(shape=[128]))

        D_W2 = tf.Variable(xavier_init([128, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_W1, D_W2, D_b1, D_b2]
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

def load(saver, sess, checkpoint_dir):
    print(" [*] Reading checkpoints...", x)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator1(z):
	with tf.variable_scope("G1"):
	    G1_W1 = tf.Variable(xavier_init([100, 100]))
        G1_b1 = tf.Variable(tf.zeros(shape=[100]))

        G1_W2 = tf.Variable(xavier_init([100, 784]))
        G1_b2 = tf.Variable(tf.zeros(shape=[784]))

        theta_G1 = [G1_W1, G1_W2, G1_b1, G1_b2]
        G1_h1 = tf.nn.relu(tf.matmul(z, G1_W1) + G1_b1)
        G1_log_prob = tf.matmul(G1_h1, G1_W2) + G1_b2
        G1_prob = tf.nn.sigmoid(G1_log_prob)

        return G1_prob

def generator2(z):
	with tf.variable_scope("G2"):
	    G2_W1 = tf.Variable(xavier_init([100, 100]))
        G2_b1 = tf.Variable(tf.zeros(shape=[100]))

        G2_W2 = tf.Variable(xavier_init([100, 784]))
        G2_b2 = tf.Variable(tf.zeros(shape=[784]))

        theta_G2 = [G2_W1, G2_W2, G2_b1, G2_b2]
        G2_h1 = tf.nn.relu(tf.matmul(z, G2_W1) + G2_b1)
        G2_log_prob = tf.matmul(G2_h1, G2_W2) + G2_b2
        G2_prob = tf.nn.sigmoid(G2_log_prob)

        return G2_prob

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

G_sample1 = generator1(Z)
G_sample2 = generator2(Z)

t_vars = tf.trainable_variables()
g1_vars = [var for var in t_vars if 'G1' in var.name]
g2_vars = [var for var in t_vars if 'G2' in var.name]
d_vars = [var for var in t_vars if 'D' in var.name]
print ([x.name for x in t_vars])

minibatch_size = 128
Z_dim = 100

saver1 = tf.train.Saver(g1_vars)
saver2 = tf.train.Saver(g2_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

load(saver2,sess,'/Users/archana/Desktop/BTP/checkpoints7/')
load(saver1,sess,'/Users/archana/Desktop/BTP/checkpoints6/')

if not os.path.exists('out8/'):
    os.makedirs('out8/')

i = 0
for it in range(1000000):
    if it % 1000 == 0:
    	p=np.random.random_sample()
    	if (p>0.5):
    		G_sample = G_sample1
    	else:
    		G_sample = G_sample2

        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out8/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
