import sys
import math
import copy
import warnings
import functools
import numpy as np
import matplotlib.pyplot as plt

from sympy.abc import x, y, z, a
from typing import Tuple, Optional
from sympy.parsing.sympy_parser import parse_expr


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


def create_hypercube_single_eps(eps, state_space, n) -> Tuple[list, list, list]:
    """
    A helper method to create hypercube intervals in each dimension using the same eps value
    """

    hypercubes_partitions = []
    hypercubes_partitions_centers = []
    partition_dimension = []

    for ii in range(n):
        ith_space = state_space[ii]
        range_per_interval = 2*eps
        length = max(ith_space) - min(ith_space)

        # For partition of the space the length should be at least 4 epsilon
        if length < 4*eps:
            print('Error: state space', ii, 'too small for partitioning over chosen epsilon:', eps)
            print('Try decreasing epsilon [0-1] or normalizing the state space')

        number_of_intervals = math.floor(length/(range_per_interval))

        x_ith_low = min(ith_space)
        r = range_per_interval
        m = int(number_of_intervals)
        partition_dimension.append(m)

        jth_hyper = []
        jth_hyper_center = []
        # Assumption: partition floating point errors for domain R negligible
        for jj in range(int(number_of_intervals)):
            x_ith_partitions = [x_ith_low + jj*r, x_ith_low + (jj+1)*r]
            jth_hyper.append(x_ith_partitions)
            jth_hyper_center.append(min(jth_hyper[jj]) + eps)

        hypercubes_partitions.append(jth_hyper)
        hypercubes_partitions_centers.append(jth_hyper_center)

    return hypercubes_partitions, hypercubes_partitions_centers, partition_dimension


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


def get_hypercube(eps, state_space, n, print_flag: bool = False):
    if isinstance(eps, float) or isinstance(eps, int):
        print("Using same Epsilon for each dimension")
        hypercubes_partitions, hypercubes_partitions_centers, partition_dimension = \
            create_hypercube_single_eps(eps, state_space, n)
    elif isinstance(eps, np.ndarray):
        print("Using variable Epsilon")
        hypercubes_partitions, hypercubes_partitions_centers, partition_dimension = \
            create_hypercube_variable_eps(eps, state_space, n)
    else:
        print("Please enter a valid type of EPS, either an array or a scalar value")
        sys.exit(-1)

    # Generate hyper matrix containing all combinations in n-dim space [recursive for loops]
    element = []
    hypermatrix = []
    dim = len(partition_dimension) - 1
    dim_count = 2
    partition_count = 0

    hypercubes = recursive_for(hypercubes_partitions, dim_count, partition_count, dim, element, hypermatrix)
    hypercubes = np.array(hypercubes)

    if print_flag:
        print("****************************************************************************************")
        print(f"The # of partitions in each dimension is {partition_dimension}")
        print("****************************************************************************************")

    return hypercubes, hypercubes_partitions


def create_hypercube_idx_dict(system_hypercubes):
    _hypercube_idx_dict: dict = {}

    # create a str of the array and add it to the dict
    for idx, cube in enumerate(system_hypercubes):
        str_cube = str(cube)
        _hypercube_idx_dict[str_cube] = idx

    return _hypercube_idx_dict


def look_up_partition(system_hypercubes,
                      system_dim_partitions,
                      system_state,
                      hypercube_dict: dict,
                      print_flag: bool = False):
    """
    A helper function to look up the hypercube the current system state belongs to

    @:param system_hypercubes: A tensor that contains an array of hypercube vertices.

    @param system_dim_partitions: The partitions in each dimension

    @param system_state: The current state of the system
    :return: the hypercube index
    """
    assert system_hypercubes[0].shape[0] == system_state.size, "Please ensure the state has the same number" \
                                                                    " of dimensions as the hypercube dimension"

    _hypercube_mat = np.zeros_like(system_hypercubes[0])
    for system_state_dim in range(system_state.size):
        curr_state = system_state[system_state_dim]

        # check the hypercube in this corresponding dim
        for _idx, dim_partition in enumerate(system_dim_partitions[system_state_dim]):
            if dim_partition[0] <= curr_state and dim_partition[1] >= curr_state:
                _hypercube_mat[system_state_dim, :] = dim_partition
                break

    if print_flag:
        print(f"Point {system_state} belong to hypercube {_hypercube_mat}")

    # now that you have the hypercube, look up the index it corresponds to in the partitions tensor
    hypercube_idx = hypercube_dict.get(str(_hypercube_mat))

    return hypercube_idx


def get_sym_poly(deg: int, poly_coeffs):
    # x[1] -> x
    # x[2] -> y
    # x[3] -> z
    # x[4] -> a
    if deg == 4:
        monomial = ["x[1]^4", "x[1]^3*x[2]", "x[1]^3*x[3]", "x[1]^3*x[4]", "x[1]^2*x[2]^2", "x[1]^2*x[2]*x[3]",
                    "x[1]^2*x[2]*x[4]", "x[1]^2*x[3]^2", "x[1]^2*x[3]*x[4]", "x[1]^2*x[4]^2", "x[1]*x[2]^3",
                    "x[1]*x[2]^2*x[3]", "x[1]*x[2]^2*x[4]", "x[1]*x[2]*x[3]^2", "x[1]*x[2]*x[3]*x[4]",
                    "x[1]*x[2]*x[4]^2", "x[1]*x[3]^3", "x[1]*x[3]^2*x[4]", "x[1]*x[3]*x[4]^2", "x[1]*x[4]^3",
                    "x[2]^4", "x[2]^3*x[3]", "x[2]^3*x[4]", "x[2]^2*x[3]^2", "x[2]^2*x[3]*x[4]", "x[2]^2*x[4]^2",
                    "x[2]*x[3]^3", "x[2]*x[3]^2*x[4]", "x[2]*x[3]*x[4]^2", "x[2]*x[4]^3", "x[3]^4", "x[3]^3*x[4]",
                    "x[3]^2*x[4]^2", "x[3]*x[4]^3", "x[4]^4", "x[1]^3", "x[1]^2*x[2]", "x[1]^2*x[3]", "x[1]^2*x[4]",
                    "x[1]*x[2]^2", "x[1]*x[2]*x[3]", "x[1]*x[2]*x[4]", "x[1]*x[3]^2", "x[1]*x[3]*x[4]", "x[1]*x[4]^2",
                    "x[2]^3", "x[2]^2*x[3]", "x[2]^2*x[4]", "x[2]*x[3]^2", "x[2]*x[3]*x[4]", "x[2]*x[4]^2", "x[3]^3",
                    "x[3]^2*x[4]", "x[3]*x[4]^2", "x[4]^3","x[1]^2", "x[1]*x[2]", "x[1]*x[3]", "x[1]*x[4]", "x[2]^2",
                    "x[2]*x[3]", "x[2]*x[4]", "x[3]^2", "x[3]*x[4]", "x[4]^2", "x[1]", "x[2]", "x[3]", "x[4]", "1"]

    elif deg == 2:
        monomial = ["x[1]^2", "x[1]*x[2]", "x[2]^2", "x[1]", "x[2]", "1"]

        assert len(monomial) == len(poly_coeffs), "Make sure you the # of coeffs is same the # of monomials in the" \
                                                  " polynomial expression"

    expr = []
    for idx, ele in enumerate(monomial):
        ele = ele.replace("^", "**")
        ele = ele.replace("x[1]", 'x')
        ele = ele.replace("x[2]", 'y')
        ele = ele.replace("x[3]", 'z')
        ele = ele.replace("x[4]", 'a')
        # multiple with its corresponding coefficient
        coeff_ele = ele + "*{}".format(poly_coeffs[idx])
        expr.append(coeff_ele)

    str_expr = '+'.join(expr)
    sym_expr = parse_expr(str_expr)
    return sym_expr


def evolve_according_to_controller(partitions,
                                   partition_dim_list,
                                   curr_state,
                                   next_state,
                                   hypercube_idx_dict,
                                   num_controllers:int,
                                   control_evolution,
                                   A_ub: np.array,
                                   A_lb: np.array,
                                   b_ub: np.array,
                                   b_lb: np.array,
                                   control_coeff_matrix: Optional,
                                   time_step: int = 0,
                                   print_flag: bool = False) -> np.array:
    """
    A helper function to get next state as per the control polynomial

    This method evolves as per x_{k+1} = f(x_{k}) + u_{k}

    :param partitions:
    :param partition_dim_list:
    :param curr_state:
    :param hypercube_idx_dict:
    :param control_coeff_matrix:
    :param print_flag:
    :return:
    """
    # write a function to look up the partition
    if len(curr_state.shape) == 2:
        curr_state = curr_state.flatten()

    if len(next_state.shape) == 2:
        next_state = next_state.flatten()

    new_state = np.copy(curr_state)
    cube_idx = look_up_partition(system_hypercubes=partitions,
                                 system_dim_partitions=partition_dim_list,
                                 system_state=curr_state,
                                 hypercube_dict=hypercube_idx_dict,
                                 print_flag=True)

    if cube_idx is None:
        print(f"**************************************ERROR at step: {time_step} ***********************************")
        import sys
        # sys.exit(-2)
        return new_state

    if num_controllers == 2 and curr_state.shape[0] == 4:
        # get the control value
        poly_coeff_1 = control_coeff_matrix[0][cube_idx]
        poly_coeff_2 = control_coeff_matrix[1][cube_idx]
        u1_poly = get_sym_poly(deg=4, poly_coeffs=poly_coeff_1)
        u2_poly = get_sym_poly(deg=4, poly_coeffs=poly_coeff_2)
        # x - cart position
        # y - dx - cart velocity
        # z - theta - cart pole angle
        # a - dtheta - cart pole angular velocity
        u1_value = u1_poly.subs([(x, curr_state[0]), (y, curr_state[1]), (z, curr_state[2]), (a, curr_state[3])])
        u2_value = u2_poly.subs([(x, curr_state[0]), (y, curr_state[1]), (z, curr_state[2]), (a, curr_state[3])])

        control_evolution[time_step, 0] = u1_value
        control_evolution[time_step, 1] = u2_value

        # add this scalar to the 2 and 4th dimension
        new_state[1] = curr_state[1] + u1_value
        new_state[3] = curr_state[3] + u2_value
    elif num_controllers == 1 and curr_state.shape[0] == 4:
        # get the control value
        poly_coeff = control_coeff_matrix[cube_idx]
        poly = get_sym_poly(deg=4, poly_coeffs=poly_coeff)
        # x - cart position
        # y - dx - cart velocity
        # z - theta - cart pole angle
        # a - dtheta - cart pole angular velocity
        value = poly.subs([(x, curr_state[0]), (y, curr_state[1]), (z, curr_state[2]), (a, curr_state[3])])
        # print(value)

        # add this scalar to the 2 and 4th dimension
        new_state[1] = curr_state[1] + value
        new_state[3] = curr_state[3] + value

    elif num_controllers == 2 and curr_state.shape[0] == 2:
        # get the control value
        poly_coeff_1 = control_coeff_matrix[0][cube_idx]
        poly_coeff_2 = control_coeff_matrix[1][cube_idx]
        u1_poly = get_sym_poly(deg=2, poly_coeffs=poly_coeff_1)
        u2_poly = get_sym_poly(deg=2, poly_coeffs=poly_coeff_2)
        # x - position
        # y - position
        u1_value = u1_poly.subs([(x, curr_state[0]), (y, curr_state[1])])
        u2_value = u2_poly.subs([(x, curr_state[0]), (y, curr_state[1])])

        # control_evolution[time_step, 0] = u1_value
        # control_evolution[time_step, 1] = u2_value

        ##### TESTING 0.5 - u case
        control_evolution[time_step, 0] = 0.5 - u1_value
        control_evolution[time_step, 1] = 0.5 - u2_value

        # add this scalar to the 2 and 4th dimension
        new_state[0] = curr_state[0] + 0.5 - u1_value
        new_state[1] = curr_state[1] + 0.5 - u2_value

    elif num_controllers == 1 and curr_state.shape[0] == 2:
        if cube_idx == 109:
            poly_coeff = control_coeff_matrix[cube_idx]
        elif cube_idx == 210:
            poly_coeff = control_coeff_matrix[cube_idx]
        else:
            print("ERROR, no controller coeff available for any other hypercubes")
            exit()

        # gamma_poly = get_sym_poly(deg=2, poly_coeffs=poly_coeff)
        # gamma_poly_value = gamma_poly.subs([(x, curr_state[0]), (y, curr_state[1])])
        #
        # u_1, u_2 = create_new_controller(gamma_poly_val=gamma_poly_value,
        #                                  A_ub=A_ub[cube_idx],
        #                                  A_lb=A_lb[cube_idx],
        #                                  b_ub=b_ub[cube_idx],
        #                                  b_lb=b_lb[cube_idx],
        #                                  curr_state=curr_state,
        #                                  next_state=next_state[0])
        #
        # new_state[0] = curr_state[0] + u_1
        # new_state[0] = curr_state[0] + u_2

        u = 0.1
        new_state[0] = next_state[0] + u
        new_state[1] = next_state[1] + u

    return new_state


def simulate_zamani_ex_1(system_state, rollout: int = 100, use_controller: bool = False):
    """
    A function to simulate zamani's Example 1 from his paper - Formal Synthesis of Stochastic Systems via control
     barrier certificates
    :return:
    """

    from sympy import poly

    def get_next_state(x, u, w) -> float:
        Ts = 5
        Th = 55
        Te = 15
        ae = 0.008
        aH = 0.0036

        f = x + Ts * (ae * (Te - x) + aH * (Th - x) * u) + (0.1 * w)
        return f

    u_poly = poly(-1.018e-6 * x**4 + 7.563e-5 * x**3 - 1.872e-3 * x**2 + 0.02022 * x + 0.3944)
    state_evolution = np.zeros(shape=(rollout + 1, ))
    control_evolution = np.zeros(shape=(rollout, ))

    curr_state = system_state
    state_evolution[0] = curr_state
    x_k = curr_state
    for step in range(rollout):
        # get next state
        if use_controller:
            u_k = u_poly.subs([(x, x_k)])
            # u_k = -1.018e-6 * x_k ** 4 + 7.563e-5 * x_k ** 3 - 1.872e-3 * x_k ** 2 + 0.02022 * x_k + 0.3944
        else:
            u_k = 0

        control_evolution[step] = u_k
        w = np.random.normal()
        x_k_prime = get_next_state(x_k, u_k, w)
        state_evolution[step+1] = x_k_prime
        x_k = x_k_prime

    fig, axs = plt.subplots(2)
    fig.suptitle('Zamani example 1')

    axs[0].plot(state_evolution, color='tab:blue')
    axs[0].axhline(y=23, color='k', linestyle='--')
    axs[0].axhline(y=20, color='k', linestyle='--')

    axs[1].plot(control_evolution, color='tab:green')
    # axs[1].axhline(y=1, color='k', linestyle='--')
    # axs[1].axhline(y=0, color='k', linestyle='--')

    plt.show(block=True)


def postprocess_partition_dump(partitions):
    # use the same dtype as the input
    transposed_mat = np.empty(shape=(partitions.shape[0], partitions.shape[2], partitions.shape[1]),
                              dtype=partitions.dtype)
    for icube in range(partitions.shape[0]):
        transposed_mat[icube] = partitions[icube].T

    return transposed_mat


def create_new_controller(gamma_poly_val: float,
                          A_ub: np.array,
                          A_lb: np.array,
                          b_ub: np.array,
                          b_lb: np.array,
                          curr_state: np.array,
                          next_state: float):
    """
    A helper function to create the controllee polynomial as per Appendix C in the paper - 04/16/22

    :return:
    """
    A_av: np.array = A_ub + A_lb
    b_av: np.array = b_ub + b_lb

    Ax: float = np.matmul(A_av[0, :], curr_state)
    Ax_b_x_prime: float = (Ax + b_av[0]) - next_state
    
    u_val_1: float = Ax_b_x_prime + math.sqrt(Ax_b_x_prime**2 - gamma_poly_val)
    u_val_2: float = Ax_b_x_prime - math.sqrt(Ax_b_x_prime ** 2 - gamma_poly_val)
    
    return u_val_1, u_val_2
    
    
    


# function to plot the learned NN's dynamics
def plot_learned_sys_phase_portrait(input_data, trained_nn_model, scale=1.0):
    """"
    A function to plot the origina phase portrait of the system
    """
    step_size = 0.5

    # if isinstance(input_data, list):
    x1 = np.array(input_data[:, 0])
    x2 = np.array(input_data[:, 1])

    max_x1 = np.amax(x1)
    min_x1 = np.amin(x1)

    max_x2 = np.amax(x2)
    min_x2 = np.amin(x2)
    # else:
    #     x1 = input_data[:, 0]
    #     x2 = input_data[:, 1]
    #     max_x1 = x1[1]
    #     min_x1 = x1[0]
    #
    #     max_x2 = x2[1]
    #     min_x2 = x2[0]

    # min_x1 = 0
    # min_x2 = 0
    # max_x1 = 10
    # max_x2 = 10
    lin_x = np.arange(min_x1, max_x1, step_size)
    lin_y = np.arange(min_x2, max_x2, step_size)

    _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(15, 15))

    for i in lin_x:
        for j in lin_y:
            curr_state = np.array([[i, j]])
            op = trained_nn_model.predict(curr_state)

            _ax.arrow(x=curr_state[0][0],
                      y=curr_state[0][1],
                      dx=scale * (op[0][0] - curr_state[0][0]),
                      dy=scale * (op[0][1] - curr_state[0][1]),
                      color='red',
                      head_width=0.1,
                      length_includes_head=True,
                      alpha=0.5)

    plt.show(block=False)

    return _fig, _ax
