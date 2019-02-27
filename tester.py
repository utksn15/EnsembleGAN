import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    with tf.variable_scope("G2"):
        G1_W1 = tf.Variable(xavier_init([100, 200]))
        G1_W2 = tf.Variable(xavier_init([200, 784]))
        G1_b1 = tf.Variable(tf.zeros(shape=[200]))
        G1_b2 = tf.Variable(tf.zeros(shape=[784]))

        theta_G1 = [G1_W1, G1_W2, G1_b1, G1_b2]
        G1_h1 = tf.nn.relu(tf.matmul(z, G1_W1) + G1_b1)
        G1_log_prob = tf.matmul(G1_h1, G1_W2) + G1_b2
        G1_prob = tf.nn.sigmoid(G1_log_prob)

        return G1_prob

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

def load(saver, sess, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print ("Succesfully loaded")
        return True
    else:
        print ("Loading Unsuccessful")
        return False

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)

t_vars = tf.trainable_variables()
g2_vars = [var for var in t_vars if 'G2' in var.name]
#g2_vars = [var for var in t_vars if 'G2' in var.name]
d_vars = [var for var in t_vars if 'D' in var.name]

#print ([x for x in t_vars])
#print d_vars

D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g2_vars)

minibatch_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

saver2 = tf.train.Saver(g2_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

load(saver2,sess,'/Users/archana/Desktop/BTP/Codes/out13/')

if not os.path.exists('out14/'):
    os.makedirs('out14/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out14/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(minibatch_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(minibatch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(minibatch_size, Z_dim)})

    saver2.save(sess, '/Users/archana/Desktop/BTP/codes/out14/', global_step=2000,write_meta_graph=False)

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
