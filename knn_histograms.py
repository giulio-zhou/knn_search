import numpy as np
import tensorflow as tf
import sys

# Bins is an array of values that denote the bucket boundaries.
def compute_knn_histograms(data_path, k, bins):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_tensor = tf.placeholder(tf.float32, shape=[None])
    norm_tensor = tf.norm(training_set_tensor - test_tensor, axis=1)

    feature_vectors = np.load(data_path)
    hist_counts = np.zeros(len(bins) - 1)
    for i, elem in enumerate(feature_vectors):
        if i % 100 == 0:
            print i
        norms = sess.run(norm_tensor,
                         feed_dict={training_set_tensor: feature_vectors,
                                    test_tensor: elem})
        hist, bin_edges = np.histogram(norms)
        hist_counts += hist
    print(hist_counts)
    return hist_counts, bins

def compute_all_pairs_histograms(data_path, bins):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_tensor = tf.placeholder(tf.float32, shape=[None])
    norm_tensor = tf.norm(training_set_tensor - test_tensor, axis=1)

    feature_vectors = np.load(data_path)
    hist_counts = np.zeros(len(bins) - 1)
    for i, elem in enumerate(feature_vectors):
        if i % 100 == 0:
            print i
        norms = sess.run(norm_tensor,
                         feed_dict={training_set_tensor: feature_vectors,
                                    test_tensor: elem})
        hist, bin_edges = np.histogram(norms)
        hist_counts += hist
    print(hist_counts)
    return hist_counts, bins

if __name__ == '__main__':
    data_path = sys.argv[1]
    k = int(sys.argv[2])
    bins = [0.5 * i for i in range(100)]
    # compute_knn_histograms(data_path, k, bins)
    compute_all_pairs_histograms(data_path, k, bins)
