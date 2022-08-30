from typing_extensions import Literal
from typing import Optional, Callable
from scipy.cluster.vq import kmeans2 as _kmeans
import numpy as np
import tensorflow as tf
from collections import namedtuple
from gpflow.config import default_float
from gpflow.utilities import to_default_int


def kmeans(training_inputs: np.ndarray, num_ip: int, max_inputs: int = 100_000):
    """
    Initialize inducing inputs using kmeans(++)

    :param training_inputs:  An array of training inputs X ⊂ 𝑋, with |X| = N < ∞. We frequently assume X= ℝ^D
    and this is [N,D]
    :param num_ip: An integer, number of inducing points to return. Equiv. "k" to use in kmeans
    :return: Z, None, M inducing inputs
    """
    num = training_inputs.shape[0]

    # normalize data
    training_inputs_stds = np.std(training_inputs, axis=0)
    if np.min(training_inputs_stds) < 1e-13:
        warnings.warn("One feature of training inputs is constant")

    training_inputs = training_inputs / training_inputs_stds
    choose_from = training_inputs
    if num > max_inputs:
        indices = np.random.choice(num, size=max_inputs, replace=False)
        choose_from = training_inputs[indices, ...]

    print(
        ">>> "
        f"The total number of points: {num}, "
        f"given the number of points to choose from: {len(choose_from)}, "
        f"and the number of inducing points: {num_ip}"
    )

    centroids, _ = _kmeans(choose_from, num_ip)
    n_centroids = len(centroids)

    # Some times K-Means returns fewer than K centroids, in this case we sample remaining point from data
    if n_centroids < num_ip:
        num_extra_points = num_ip - n_centroids
        indices = np.random.choice(num, size=num_extra_points, replace=False)
        additional_points = training_inputs[indices]
        centroids = np.concatenate([centroids, additional_points], axis=0)

    return centroids * training_inputs_stds, None


def make_inducing_selection_function(
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    compile: Optional[Literal["xla"]] = None,
):
    use_n_factor = False
    use_jit = True if compile == "xla" else False

    def select(datapoints, numips, sigma_sq, threshold):
        iv, other = greedy_selection_tf(
            datapoints, numips, kernel, sigma_sq, threshold, use_n_factor=use_n_factor
        )
        return iv, other

    select_tf = tf.function(select, jit_compile=use_jit)

    def selection_from_subset_func(
        datapoints: np.array,
        subset_size: int,
        numips: int,
        sigma_sq: float,
        threshold: float,
        rng: Optional[np.random.RandomState] = None,
    ):
        if rng is None:
            rng = np.random
        dataset_size = datapoints.shape[0]
        subset_size = np.min([dataset_size, subset_size])
        subset_indices = rng.choice(dataset_size, size=subset_size, replace=False)
        subset_mask = np.array([False] * dataset_size)
        subset_mask[subset_indices] = True
        datapoints = np.array(datapoints)
        subset = datapoints[subset_mask]
        iv, other = select_tf(subset, numips, sigma_sq, threshold)
        return iv.numpy(), other.numpy(), subset_mask

    return selection_from_subset_func


def run_uniform_greedy_selection(
    selection_func,
    datapoints: np.array,
    subset_size: int,
    numips: int,
    sigma_sq: float,
    threshold: float,
    rng: Optional[np.random.RandomState] = None,
):
    iv, other, subset_mask = selection_func(
        datapoints, subset_size, numips, sigma_sq, threshold, rng=rng
    )
    iv = np.array(iv)
    if iv.shape[0] < numips:
        import warnings

        warn_msg = (
            f"Greedy method with threshold {threshold} selected less inducing points "
            f"than the requested maximum ({iv.shape[0]} < {numips})"
        )
        warnings.warn(warn_msg, UserWarning)

        fill_size = numips - iv.shape[0]
        invert_subset_mask = np.invert(subset_mask)
        invert_subset = datapoints[invert_subset_mask]
        fill_indices = rng.choice(invert_subset.shape[0], size=fill_size, replace=False)
        fill_elements = invert_subset[fill_indices]
        iv = np.concatenate([iv, fill_elements], axis=0)
    return iv, other


def uniform_greedy_selection(
    training_inputs: np.ndarray,
    subset_size: int,
    numips: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    sigma_sq: float,
    threshold: float,
    rng: np.random.RandomState = None,
):
    if rng is None:
        rng = np.random
    dataset_size = training_inputs.shape[0]
    subset_size = np.min([dataset_size, subset_size])
    subset_indices = rng.choice(dataset_size, size=subset_size, replace=False)
    subset_mask = np.array([False] * dataset_size)
    subset_mask[subset_indices] = True
    training_inputs = np.array(training_inputs)
    subset = training_inputs[subset_mask]
    iv, other = greedy_selection_tf(subset, numips, kernel, sigma_sq, threshold, use_n_factor=False)

    if iv.shape[0] < numips:
        import warnings

        warn_msg = (
            f"Greedy method with threshold {threshold} selected less inducing points "
            f"than the requested maximum ({iv.shape[0]} < {numips})"
        )
        warnings.warn(warn_msg, UserWarning)

        fill_size = numips - iv.shape[0]
        invert_subset_mask = np.invert(subset_mask)
        invert_subset = training_inputs[invert_subset_mask]
        fill_indices = rng.choice(invert_subset.shape[0], size=fill_size, replace=False)
        fill_elements = invert_subset[fill_indices]
        iv = np.concatenate([iv, fill_elements], axis=0)
    return iv, other


def greedy_selection(
    training_inputs: np.ndarray,
    M: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    sigma_sq: float,
    threshold: float,
    use_n_factor: bool = True,
    rng: np.random.RandomState = None,
):
    if rng is None:
        rng = np.random
    training_inputs = np.array(training_inputs)
    N = training_inputs.shape[0]
    perm = rng.permutation(N)  # permute entries so tiebreaking is random
    training_inputs = training_inputs[perm]
    # note this will throw an out of bounds exception if we do not update each entry
    indices = np.zeros(M, dtype=int) + N
    di = kernel(training_inputs, None, full_cov=False) + 1e-12  # jitter

    # if self.sample:
    # indices[0] = sample_discrete(di) TODO(awav)
    if False:
        pass
    else:
        indices[0] = np.argmax(di)  # select first point, add to index 0
    if M == 1:
        indices = indices.astype(int)
        Z = training_inputs[indices]
        indices = perm[indices]
        return Z, indices

    ci = np.zeros((M - 1, N))  # [M,N]

    for m in range(M - 1):
        j = int(indices[m])  # int
        new_Z = training_inputs[j : j + 1]  # [1,D]
        dj = np.sqrt(di[j])  # float
        cj = ci[:m, j]  # [m, 1]
        Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
        L = np.round(np.squeeze(Lraw), 20)  # [N]
        L[j] += 1e-12  # jitter
        ei = (L - np.dot(cj, ci[:m])) / dj
        ci[m, :] = ei
        try:
            di -= ei ** 2
        except FloatingPointError:
            pass
        di = np.clip(di, 0, None)

        # if self.sample:
        # indices[m + 1] = sample_discrete(di) TODO(awav)
        if False:
            pass
        else:
            indices[m + 1] = np.argmax(di)  # select first point, add to index 0
        # sum of di is tr(Kff-Qff), if this is small things are ok
        trace = np.sum(np.clip(di, 0, None))
        trace /= (sigma_sq * N) if use_n_factor else sigma_sq
        if trace < threshold:  # TODO(awav)
            indices = indices[:m]
            break

    indices = indices.astype(int)
    Z = training_inputs[indices]
    indices = perm[indices]
    return Z, indices


def greedy_selection_tf(
    training_inputs: tf.Tensor,
    M: int,
    kernel: Callable[[tf.Tensor, Optional[tf.Tensor], Optional[bool]], tf.Tensor],
    sigma_sq: float = 1.0,
    threshold: float = 0.0,
    use_n_factor: bool = True,
):
    N = tf.shape(training_inputs)[0]
    print(f"[greedy_selection_tf] N={N}, M={M}")
    M = N if M > N else M

    perm = tf.random.shuffle(tf.range(N))
    X = tf.gather(training_inputs, perm)  # shuffle training data
    di = kernel(X, full_cov=False)  #  diagonal entries of kernel
    inds = tf.math.argmax(di)[None]  # select first point, point with highest variance
    ci = tf.zeros([1, N], dtype=default_float())

    State = namedtuple("State", "m, inds, di, ci")
    size = tf.cast(N, di.dtype)

    def stopping_criterion(state):
        trace = tf.reduce_sum(state.di) / sigma_sq
        trace_is_small = (trace / size if use_n_factor else trace) <= threshold
        too_many_ips = state.m >= M
        return not (trace_is_small or too_many_ips)

    def loop_body(state):
        j = to_default_int(state.inds[-1])  # index in X of last point chose
        new_Z = X[j : j + 1]  # input value of last point chosen
        dj = tf.math.sqrt(state.di[j])  # conditional standard deviation of point chosen
        cj = state.ci[: state.m, j : j + 1]  # [m, 1]
        K = kernel(X, new_Z, full_cov=True)  # [n, 1],  covariance between new point and all points
        ei = (K - tf.matmul(state.ci, cj, transpose_a=True)) / dj  # [n,1]
        new_ci = tf.concat([state.ci, tf.transpose(ei)], axis=0)
        new_di = state.di - tf.square(ei)[:, 0]
        new_inds = tf.concat([state.inds, [tf.math.argmax(new_di)]], axis=0)
        new_state = State(state.m + 1, new_inds, new_di, new_ci)
        return [new_state]

    init_state = [State(1, inds, di, ci)]
    tshape = tf.TensorShape

    shape_invariants = [State(tshape([]), tshape([None]), tshape([None]), tshape([None, None]))]

    final_state = tf.while_loop(
        stopping_criterion, loop_body, init_state, shape_invariants=shape_invariants
    )
    final_state = tf.nest.map_structure(tf.stop_gradient, final_state)
    perm_inds = tf.gather(perm, final_state[0].inds)
    Z = tf.gather(training_inputs, perm_inds)
    return Z, perm_inds
