from networks_exp_utils.file_util import LabelIterator
from networks_exp_utils.file_util import ShardedFileIterator
import numpy as np
import tensorflow as tf
import sys

def knn_strategy(data_path, k, d, thresh, start_budget):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_tensor = tf.placeholder(tf.float32, shape=[None])
    norm_tensor = tf.norm(training_set_tensor - test_tensor, axis=1)
    top_k_vals_tensor, top_k_idx_tensor = tf.nn.top_k(norm_tensor, k)
    num_below_thresh_tensor = \
        tf.reduce_sum(tf.cast(top_k_vals_tensor < thresh, tf.int32))

    data_iterator = ShardedFileIterator(data_path)
    training_set = data_iterator.get_next_entries(start_budget)
    for i in range(data_iterator.get_data_len() - start_budget):
        elem = data_iterator.get_next_entries(1).squeeze()
        if i % 10 == 0:
            print i
        num_below_thresh = sess.run(
            num_below_thresh_tensor,
            feed_dict={training_set_tensor: training_set, test_tensor: elem})
        if num_below_thresh < d:
        # if np.sum(norms[top_k_idx] < thresh) < d:
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
    # Separate computation graph into first part that computes all pairwise
    # norms and the second that just computes the top k. This allows us to
    # split up the first part into sub batches to fit into GPU memory and 
    # process the second half sequentially.
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    norm_tensor = compute_pairwise_dists(training_set_tensor, test_set_tensor)
    norm_tensor = tf.transpose(norm_tensor)
    norm_tensor_placeholder = tf.placeholder(tf.float32, shape=[None])
    top_k_vals_tensor, top_k_idx_tensor = tf.nn.top_k(-norm_tensor_placeholder, k)
    top_k_vals_tensor = -top_k_vals_tensor
    num_below_thresh_tensor = \
        tf.reduce_sum(tf.cast(top_k_vals_tensor < thresh, tf.int32))

    data_iterator = ShardedFileIterator(data_path)
    training_set = data_iterator.get_next_entries(start_budget)
    training_set_indices = [i for i in range(len(training_set))]

    num_test_examples = data_iterator.get_data_len() - start_budget
    num_batches = num_test_examples // batch_size
    remainder = num_test_examples % batch_size
    max_norm_batch_size = 10000
    for i in range(num_batches):
        if i % 10 == 0:
            print i, len(training_set)
        elems = data_iterator.get_next_entries(batch_size).squeeze()
        elems_idx = start_budget + i * batch_size + np.arange(batch_size)
        num_full_norm_batches = len(training_set) // max_norm_batch_size
        norm_batch_remainder = len(training_set) % max_norm_batch_size
        norms = []
        for k in range(num_full_norm_batches):
            training_set_slice = training_set[k*max_norm_batch_size:
                                              (k+1)*max_norm_batch_size]
            norm_slice = sess.run(norm_tensor,
                                  feed_dict={training_set_tensor: training_set_slice,
                                             test_set_tensor: elems})
            norms.append(norm_slice)
        if norm_batch_remainder > 0:
            training_set_slice = training_set[-norm_batch_remainder:]
            norm_slice = sess.run(norm_tensor,
                                  feed_dict={training_set_tensor: training_set_slice,
                                             test_set_tensor: elems})
            norms.append(norm_slice)
        # Compute pairwise distances among test examples.
        test_pairwise_norms = sess.run(norm_tensor,
                                       feed_dict={training_set_tensor: elems,
                                                  test_set_tensor: elems})
        # Concatenate norms and compute num_below_thresh.
        composite_norm_npy = np.hstack(norms)
        new_samples = []
        test_set_collected_samples_idx = []
        for j, training_norms in enumerate(composite_norm_npy):
            test_norms = test_pairwise_norms[j, test_set_collected_samples_idx]
            num_below_thresh = sess.run(
                num_below_thresh_tensor,
                feed_dict={norm_tensor_placeholder: np.append(training_norms, test_norms)})
            if num_below_thresh < d:
                new_samples.append(elems[j])
                training_set_indices.append(elems_idx[j])
                test_set_collected_samples_idx.append(j)
        if len(new_samples) > 0:
            training_set = np.vstack([training_set, np.array(new_samples)])
    return training_set_indices

def compute_top_k_dists(X, Z, k, batch_size):
    dist_vectors = []
    dist_indices = []
    for i in range(batch_size):
        top_k_vals, top_k_idx = tf.nn.top_k(tf.norm(X - Z[i], axis=1), k)
        dist_vectors.append(top_k_vals)
        dist_indices.append(top_k_idx)
    stacked_dist_vectors = tf.transpose(tf.stack(dist_vectors))
    stacked_dist_indices = tf.transpose(tf.stack(dist_indices))
    return stacked_dist_vectors, stacked_dist_indices

def knn_strategy_batched_split(data_path, k, d, thresh, start_budget, batch_size):
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_set_tensor = tf.placeholder(tf.float32, shape=[batch_size, None])
    top_k_vals_tensor, top_k_idx_tensor = \
        compute_top_k_dists(training_set_tensor, test_set_tensor, k, batch_size)
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
        top_k_vals, num_below_thresh = sess.run(
            [top_k_vals_tensor, num_below_thresh_tensor],
            feed_dict={training_set_tensor: training_set, test_set_tensor: elems})
        print(top_k_vals)
        new_samples = []
        for j, num_below in enumerate(num_below_thresh):
            if num_below < d:
                new_samples.append(elems[j])
        if len(new_samples) > 0:
            print('added %d new samples to training set' % len(new_samples))
            training_set = np.vstack([training_set, np.array(new_samples)])

def compute_y_means(training_idx, start_budget, batch_size, data_path):
    data_iterator = ShardedFileIterator(data_path)
    data_length = data_iterator.get_data_len()
    num_batches = (data_length - start_budget) // batch_size
    remainder = (data_length - start_budget) % batch_size
    labels = LabelIterator(data_path)
    labels = labels.get_next_entries(labels.get_data_len())
    training_idx = np.array(training_idx)

    ones_count = np.sum(labels[:start_budget])
    zeros_count = start_budget - ones_count
    y_means = [ones_count / float(start_budget)]
    for i in range(num_batches):
        start = start_budget + i * batch_size
        end = start_budget + (i + 1) * batch_size
        ys_idx = training_idx[np.where((training_idx >= start) & (training_idx < end))]
        ys = labels[ys_idx]
        ones_count += np.sum(ys)
        zeros_count += len(ys) - np.sum(ys)
        y_means.append(ones_count / float(ones_count + zeros_count))
    if remainder > 0:
        offset = start_budget + num_batches * batch_size
        ys_idx = training_idx[np.where((training_idx >= start) & (training_idx < end))]
        ys = labels[ys_idx]
        ones_count += np.sum(ys)
        zeros_count += len(ys) - np.sum(ys)
        y_means.append(ones_count / float(ones_count + zeros_count))
    return y_means

def sweep_knn_strategies(data_path, output_path):
    k = 20
    d_vals = [5, 10]
    thresh_vals = [4]
    # d_vals = [5, 10]
    # thresh_vals = [5, 6]
    # d_vals = [5, 10, 20, 50]
    # thresh_vals = [5, 6, 7]
    start_budget = 2000
    batch_size = 512
    for d in d_vals:
        for thresh in thresh_vals:
            training_idx = knn_strategy_batched(
                data_path, k, d, thresh, start_budget, batch_size)
            # Compute mean of y's over time.
            y_means = compute_y_means(training_idx, start_budget,
                                      batch_size, data_path)
            with open(output_path, 'a') as outfile:
                outfile.write(''.join(['='] * 30 + ['\n']))
                # outfile.write('k: %d\n' % k)
                outfile.write('d: %d\n' % d)
                outfile.write('thresh: %d\n' % thresh)
                # outfile.write('start_budget: %d\n' % start_budget)
                # outfile.write('batch_size: %d\n' % batch_size)
                outfile.write('num indices picked: %d\n' % len(training_idx))
                outfile.write('indices picked: %s\n' % str(training_idx))
                outfile.write('y means: %s\n' % str(y_means))

def knn_strategy_entropy_targeted_batched(
        data_path, start_budget, batch_size,
        sigma, lmda, y_target, delta_benefit_thresh):
    # Separate computation graph into first part that computes all pairwise
    # norms and the second that computes the gaussian kernels. This allows us to
    # split up the first part into sub batches to fit into GPU memory and 
    # process the second half sequentially.
    sess = tf.Session()
    training_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    test_set_tensor = tf.placeholder(tf.float32, shape=[None, None])
    norm_tensor = compute_pairwise_dists(training_set_tensor, test_set_tensor)
    norm_tensor = tf.transpose(norm_tensor)
    norm_tensor_placeholder = tf.placeholder(tf.float32, shape=[None, None])
    gaussian_kernel_tensor = tf.exp(-tf.square(norm_tensor_placeholder) / sigma)

    data_iterator = ShardedFileIterator(data_path)
    training_set = data_iterator.get_next_entries(start_budget)
    training_set_indices = [i for i in range(len(training_set))]
    labels = LabelIterator(data_path)
    labels = labels.get_next_entries(labels.get_data_len())
    label_set = labels[:start_budget]

    num_test_examples = data_iterator.get_data_len() - start_budget
    num_batches = num_test_examples // batch_size
    remainder = num_test_examples % batch_size
    max_norm_batch_size = 10000
    for i in range(num_batches):
        if i % 10 == 0:
            print i, len(training_set)
        elems = data_iterator.get_next_entries(batch_size).squeeze()
        elems_idx = start_budget + i * batch_size + np.arange(batch_size)
        num_full_norm_batches = len(training_set) // max_norm_batch_size
        norm_batch_remainder = len(training_set) % max_norm_batch_size
        norms = []
        for k in range(num_full_norm_batches):
            training_set_slice = training_set[k*max_norm_batch_size:
                                              (k+1)*max_norm_batch_size]
            norm_slice = sess.run(norm_tensor,
                                  feed_dict={training_set_tensor: training_set_slice,
                                             test_set_tensor: elems})
            norms.append(norm_slice)
        if norm_batch_remainder > 0:
            training_set_slice = training_set[-norm_batch_remainder:]
            norm_slice = sess.run(norm_tensor,
                                  feed_dict={training_set_tensor: training_set_slice,
                                             test_set_tensor: elems})
            norms.append(norm_slice)
        # Compute pairwise distances among test examples.
        test_pairwise_norms = sess.run(norm_tensor,
                                       feed_dict={training_set_tensor: elems,
                                                  test_set_tensor: elems})
        # Concatenate norms and compute gaussian kernels.
        composite_norm_npy = np.hstack(norms)
        training_set_gaussian_kernels = sess.run(
            gaussian_kernel_tensor, 
            feed_dict={norm_tensor_placeholder: composite_norm_npy})
        test_set_gaussian_kernels = sess.run(
            gaussian_kernel_tensor, 
            feed_dict={norm_tensor_placeholder: test_pairwise_norms})
        # Process each sample sequentially using procedure.
        new_labels = []
        new_samples = []
        test_set_collected_samples_idx = []
        y_new_batch = labels[elems_idx]
        for j, training_kernel_vals in enumerate(training_set_gaussian_kernels):
            y_new = y_new_batch[j]
            test_kernel_vals = test_set_gaussian_kernels[j, test_set_collected_samples_idx]
            all_kernel_vals = np.append(training_kernel_vals, test_kernel_vals)
            normalized_kernel_vals = all_kernel_vals / np.sum(all_kernel_vals)
            augmented_label_set = np.append(label_set, new_labels)
            y_hat = np.sum(normalized_kernel_vals * augmented_label_set)
            y_bar = np.mean(augmented_label_set)
            y_bar_shift = 0.5 + min(0.5, y_bar - y_target) # Balance class labels.
            N = float(len(all_kernel_vals))
            delta_benefit = -np.sum(all_kernel_vals) + N**2 * lmda * (y_hat - 0.5) * np.log((1 - y_bar_shift) / y_bar_shift)
            if delta_benefit > delta_benefit_thresh:
                new_samples.append(elems[j])
                new_labels.append(y_new)
                training_set_indices.append(elems_idx[j])
                test_set_collected_samples_idx.append(j)
        if i % 10 == 0:
            print('y_bar: %f', y_bar)
            print('y_bar_shift: %f', y_bar_shift)
            print(-np.sum(all_kernel_vals), N**2 * lmda * (y_hat - 0.5) * np.log((1 - y_bar_shift) / y_bar_shift))
        if len(new_samples) > 0:
            training_set = np.vstack([training_set, np.array(new_samples)])
            label_set = np.hstack([label_set, np.array(new_labels)])
    return training_set_indices

def sweep_knn_entropy_targeted_strategies(data_path, output_path):
    start_budget = 2000
    batch_size = 512
    y_target = 0.1
    sigma_vals = [10]
    lmda_vals = [1e-4]
    delta_benefit_thresh_vals = [-100]
    for lmda in lmda_vals:
        for sigma in sigma_vals:
            for delta_benefit_thresh in delta_benefit_thresh_vals:
                training_idx = knn_strategy_entropy_targeted_batched(
                    data_path, start_budget, batch_size, sigma, lmda,
                    y_target, delta_benefit_thresh)
                # Compute mean of y's over time.
                y_means = compute_y_means(training_idx, start_budget,
                                          batch_size, data_path)
                with open(output_path, 'a') as outfile:
                    outfile.write(''.join(['='] * 30 + ['\n']))
                    # outfile.write('start_budget: %d\n' % start_budget)
                    # outfile.write('batch_size: %d\n' % batch_size)
                    outfile.write('sigma: %f\n' % sigma)
                    outfile.write('lmda: %f\n' % lmda)
                    outfile.write('delta_benefit_thresh: %f\n' % delta_benefit_thresh)
                    outfile.write('num indices picked: %d\n' % len(training_idx))
                    outfile.write('indices picked: %s\n' % str(training_idx))
                    outfile.write('y means: %s\n' % str(y_means))



if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    # knn_strategy(data_path, k, d, thresh, start_budget)
    # knn_strategy_batched(data_path, k, d, thresh, start_budget, batch_size)
    # knn_strategy_batched_split(data_path, k, d, thresh, start_budget, batch_size)
    # sweep_knn_strategies(data_path, output_path)

    sweep_knn_entropy_targeted_strategies(data_path, output_path)
