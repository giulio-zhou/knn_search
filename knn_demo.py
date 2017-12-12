from networks_exp_utils.file_util import ShardedFileIterator
import numpy as np
import tensorflow as tf
import sys

data_path = sys.argv[1]
k = 30
d = 10
thresh = 10
start_budget = 2000
batch_size = 256

def knn_strategy(data_path, k, d, thresh, start_budget):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_tensor = tf.placeholder(tf.float32, shape=[None])
    norm_tensor = tf.norm(training_set_tensor - test_tensor, axis=1)
    top_k_tensor = tf.nn.top_k(norm_tensor, k)

    data_iterator = ShardedFileIterator(data_path)
    training_set = data_iterator.get_next_entries(start_budget)
    for i in range(data_iterator.get_data_len() - start_budget):
        elem = data_iterator.get_next_entries(1).squeeze()
        if i % 10 == 0:
            print i
        norms, (top_k_vals, top_k_idx) = sess.run(
            [norm_tensor, top_k_tensor],
            feed_dict={training_set_tensor: training_set, test_tensor: elem})
        if np.sum(norms[top_k_idx] < thresh) < d:
            # Add elem in the training set
            print('add', i)
            training_set = np.append(training_set, elem.reshape(1, -1), axis=0)

def compute_pairwise_dists(X, Z):
    num_X = tf.shape(X)[0]
    num_Z = tf.shape(Z)[0]
    X_squared_norm = tf.square(tf.norm(X, axis=1))
    Z_squared_norm = tf.square(tf.norm(Z, axis=1))
    cross_terms = tf.matmul(X, tf.transpose(Z))
    D = tf.add(-2 * cross_terms, Z_squared_norm)
    D = tf.add(X_squared_norm, tf.transpose(D))
    D = tf.sqrt(tf.transpose(D))
    return D

def knn_strategy_batched(data_path, k, d, thresh, start_budget, batch_size):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_set_tensor = tf.placeholder(tf.float32, shape=[batch_size, None])
    norm_tensor = compute_pairwise_dists(training_set_tensor, test_set_tensor)
    top_k_vals_tensor, top_k_idx_tensor = tf.nn.top_k(tf.transpose(norm_tensor), k)
    num_below_thresh_tensor = \
        tf.reduce_sum(tf.cast(top_k_vals_tensor < thresh, tf.int32), axis=1)

    data_iterator = ShardedFileIterator(data_path)
    training_set = data_iterator.get_next_entries(start_budget)

    num_test_examples = data_iterator.get_data_len() - start_budget
    num_batches = num_test_examples // batch_size
    remainder = num_test_examples % batch_size
    for i in range(num_batches):
        print i, len(training_set)
        elems = data_iterator.get_next_entries(batch_size).squeeze()
        num_below_thresh = sess.run(
            num_below_thresh_tensor,
            feed_dict={training_set_tensor: training_set, test_set_tensor: elems})
        new_samples = []
        for j, num_below in enumerate(num_below_thresh):
            if num_below < d:
                new_samples.append(elems[j])
        if len(new_samples) > 0:
            print('added %d new samples to training set' % len(new_samples))
            training_set = np.vstack([training_set, np.array(new_samples)])

# knn_strategy(data_path, k, d, thresh, start_budget)
knn_strategy_batched(data_path, k, d, thresh, start_budget, batch_size)
