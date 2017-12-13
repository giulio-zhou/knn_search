import numpy as np
import tensorflow as tf
import sys

def compute_knn_histograms(data_path, k):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_tensor = tf.placeholder(tf.float32, shape=[None])
    norm_tensor = tf.norm(training_set_tensor - test_tensor, axis=1)
    top_k_vals_tensor, top_k_idx_tensor = tf.nn.top_k(-norm_tensor, k)
    top_k_vals_tensor = -top_k_vals_tensor

    feature_vectors = np.load(data_path)
    knn_dists = []
    for i, elem in enumerate(feature_vectors):
        if i % 100 == 0:
            print i
        top_k_vals = sess.run(top_k_vals_tensor,
                              feed_dict={training_set_tensor: feature_vectors,
                                         test_tensor: elem})
        knn_dists.append(top_k_vals)
    knn_dists = np.array(knn_dists)
    hist = np.histogram(knn_dists)
    return hist

if __name__ == '__main__':
    data_path = sys.argv[1]
    k = int(sys.argv[2])
    compute_knn_histograms(data_path, k)
