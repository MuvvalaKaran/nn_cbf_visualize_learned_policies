import gym
import math
import numpy as np
import tensorflow as tf

from sympy import *
from sympy import polys
from sympy.abc import x, y, z, a
from sympy.parsing.sympy_parser import parse_expr

from gen_utls import recursive_for
from gen_utls import create_hypercube_variable_eps
from scipy.io import loadmat

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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



def look_up_partition(system_hypercubes, system_dim_partitions, system_state, hypercube_dict: dict):
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

    # print(_hypercube_mat)

    # now that you have the hypercube, look up the index it corresponds to in the partitions tensor
    hypercube_idx = hypercube_dict.get(str(_hypercube_mat))

    return hypercube_idx



def _create_hypercube_idx_dict(system_hypercubes):
    _hypercube_idx_dict: dict = {}

    # create a str of the array and add it to the dict
    for idx, cube in enumerate(system_hypercubes):
        str_cube = str(cube)
        _hypercube_idx_dict[str_cube] = idx

    return _hypercube_idx_dict



def get_hypercube(eps, state_space, n, print_flag: bool = False):
    hypercubes_partitions, hypercubes_partitions_centers, partition_dimension = \
        create_hypercube_variable_eps(eps.flatten(), state_space, n)

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


def evolve_according_to_controller(partitions,
                                   partition_dim_list,
                                   curr_state,
                                   hypercube_idx_dict,
                                   control_coeff_matrix: np.array,
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

    new_state = np.copy(curr_state)
    cube_idx = look_up_partition(system_hypercubes=partitions,
                                 system_dim_partitions=partition_dim_list,
                                 system_state=curr_state,
                                 hypercube_dict=hypercube_idx_dict)

    if cube_idx is None:
        return new_state

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

    return new_state




def simulate_cartpole_w_cbf(system_state: np.array, mat_file_dir: str, rollout: int = 100, print_flag: bool = False):
    """
    A helper function to simulate the cartpole behavior w cbf
    :return:
    """
    cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_models/" \
                   "cartpole_model_1_cross_128_loss_0.01_data_2500"

    steps: int = 0

    # load the file
    control_dict = loadmat(mat_file_dir)
    eps = control_dict.get("eps")
    state_space = control_dict.get("state_space")
    s_dim = state_space.shape[0]  # System dimension
    control_coeff_matrix = control_dict.get('control_matrix')

    partitions, partition_dim_list = get_hypercube(eps=eps, state_space=state_space, n=s_dim, print_flag=print_flag)

    # create the hypercube lookup dict
    hypercube_idx_dict = _create_hypercube_idx_dict(partitions)

    # write a function to look up the partition
    # cube_idx = look_up_partition(system_hypercubes=partitions,
    #                              system_dim_partitions=partition_dim_list,
    #                              system_state=system_state,
    #                              hypercube_dict=hypercube_idx_dict)

    # # get the control value
    # poly_coeff = control_coeff_matrix[cube_idx]

    # poly = get_sym_poly(deg=4, poly_coeffs=poly_coeff)
    # value = poly.subs([(x, 1), (y, 1), (z, 1), (a, 1)])
    # print(value)

    # rollout trajectory
    cartpole = gym.make("CartPole-v0")
    cartpole_model = tf.keras.models.load_model(cartpole_dir)
    cartpole.reset()
    cartpole.render()

    use_controller: bool = True

    # if record_flag:
    #     vid = video_recorder.VideoRecorder(gym_env,
    #                                        path='/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/NN_videos/video_5.mp4')

    # start simulation
    curr_state = system_state.reshape(1, 4)
    # evolve_according_to_controller(partitions=partitions,
    #                                partition_dim_list=partition_dim_list,
    #                                curr_state=curr_state,
    #                                hypercube_idx_dict=hypercube_idx_dict,
    #                                control_coeff_matrix=control_coeff_matrix,
    #                                print_flag=False)

    for steps in range(rollout):
        # get next state
        if use_controller:
            tmp_next_state = cartpole_model.predict(curr_state)
            next_state = evolve_according_to_controller(partitions=partitions,
                                                        partition_dim_list=partition_dim_list,
                                                        curr_state=tmp_next_state,
                                                        hypercube_idx_dict=hypercube_idx_dict,
                                                        control_coeff_matrix=control_coeff_matrix,
                                                        print_flag=False)

        else:
            next_state = cartpole_model.predict(curr_state)
        cartpole.unwrapped.state = next_state.reshape(4, )
        # if record_flag:
        #     vid.capture_frame()
        cartpole.render()
        curr_state = next_state.reshape(1, 4)

    cartpole.close()




if __name__ == "__main__":
    cartpole_control_mat_file = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
                                "pendulum_control_data/control_lags_cartpole_960.mat"

    _system_state = np.array([0, 0, -1*math.pi/180, 0])

    simulate_cartpole_w_cbf(system_state=_system_state,
                            mat_file_dir=cartpole_control_mat_file,
                            print_flag=True, rollout=500)
