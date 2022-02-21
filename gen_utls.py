import math
import copy
import warnings
import functools

from typing import Tuple


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def create_hypercube_variable_eps(eps, state_space, n) -> Tuple[list, list, list]:
    """
        A helper method to create hypercube intervals in each dimension using the variable eps value
    """

    if eps.shape[0] != state_space.shape[0]:
        print("Please ensure that the epsilon is specified for each dimension")

    hypercubes_partitions = []
    hypercubes_partitions_centers = []
    partition_dimension = []

    for ii in range(n):
        ith_space = state_space[ii]
        range_per_interval = 2 * eps[ii]
        length = max(ith_space) - min(ith_space)

        # For partition of the space the length should be at least 4 epsilon
        if length < 4 * eps[ii]:
            print('Error: state space', ii, 'too small for partitioning over chosen epsilon:', eps)
            print('Try decreasing epsilon [0-1] or normalizing the state space')

        number_of_intervals = math.floor(length / (range_per_interval))

        x_ith_low = min(ith_space)
        r = range_per_interval
        m = int(number_of_intervals)
        partition_dimension.append(m)

        jth_hyper = []
        jth_hyper_center = []
        # Assumption: partition floating point errors for domain R negligible
        for jj in range(int(number_of_intervals)):
            x_ith_partitions = [x_ith_low + jj * r, x_ith_low + (jj + 1) * r]
            jth_hyper.append(x_ith_partitions)
            jth_hyper_center.append(min(jth_hyper[jj]) + eps[ii])

        hypercubes_partitions.append(jth_hyper)
        hypercubes_partitions_centers.append(jth_hyper_center)

    return hypercubes_partitions, hypercubes_partitions_centers, partition_dimension


def recursive_for(hypercube_partitions, dim_count, partition_count, dim, element, hypermatrix):

    for zz in range(len(hypercube_partitions[partition_count])):
        element1 = copy.deepcopy(element)
        element1.append(hypercube_partitions[partition_count][zz])

        if partition_count < dim:
            partition_count += 1
            recursive_for(hypercube_partitions, dim_count, partition_count, dim, element1, hypermatrix)
            partition_count -= 1
        else:
            element2 = copy.deepcopy(element1)
            hypermatrix.append(element2)
            element1.remove(element1[-1])

    return hypermatrix
