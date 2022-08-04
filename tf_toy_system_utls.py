import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from gen_utls import create_hypercube_idx_dict, evolve_according_to_controller, \
    postprocess_partition_dump, get_hypercube, plot_learned_sys_phase_portrait

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def plot_2d_toy_sys_from_data_file(data_array: np.array,
                                   file_path: str = '',
                                   ylabels: list = [],
                                   title: str = '',
                                   plot_boundary: bool = False,
                                   bdry_dict: dict = {},
                                   block_plot: bool = False):
    """
    A helper function to plot using the data file.
    :param file_path:
    :return:
    """

    assert len(data_array.shape) == 2, "Make sure the data matrix is a 2d matrix"
    assert data_array.shape[1] == len(ylabels), "Make sure you have correct # of ylabels"

    if data_array is None:
        with open(file_path, 'rb') as file_handle:
            data_array = np.load(file_handle,  allow_pickle=True)

    state_dim = data_array.shape[1]
    my_dpi = 106
    fig, axs = plt.subplots(state_dim, figsize=(660 / my_dpi, 520 / my_dpi), dpi=my_dpi)

    # figsize = (560 / my_dpi, 420 / my_dpi), dpi = my_dpi
    # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(15, 15))  # og command
    # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'), figsize=(500 / my_dpi,  / my_dpi), dpi=my_dpi)  # og command
    # _fig, _ax = plt.subplots(figsize=(560 / my_dpi, 420 / my_dpi), dpi=my_dpi)  # og command
    # fig.suptitle('Cartpole Model - Trajectory in each dimension')
    # fig.suptitle('Pendulum Model - Trajectory in each dimension')
    fig.suptitle(title)

    # ylabels: list = [r"$x (m)$", r"$y (m)$"]
    # ylabels: list = [r"$\theta (rad)$", r"$\theta dot (rad/s)$"]
    color = ['tab:blue']
    xyloc: tuple = (-20, 20)

    for ax_id in range(state_dim):
        # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        # axs idexing is not supported where there is only one plot
        if state_dim > 1:
            # axs[ax_id].plot(data_array[:, ax_id] * 180 / math.pi, color[0], label=ylabels[ax_id])  # converting to rad
            # if ax_id == 2 or ax_id == 3:
            #     axs[ax_id].plot(data_array[:, ax_id] * 180/math.pi, color[0], label=ylabels[ax_id]) # converting to rad
            # else:
            axs[ax_id].plot(data_array[:, ax_id], color[0], label=ylabels[ax_id])
            # if ax_id == 3:
            # axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
            axs[ax_id].set(ylabel=ylabels[ax_id])
            # axs[ax_id].legend(loc='best')
            axs[ax_id].grid()
            # axs.set_xticks([])
        else:
            axs.plot(data_array[:, ax_id], color[0], label=ylabels[ax_id])
            axs.set(xlabel='time-step', ylabel=ylabels[ax_id])
            axs.legend(loc='best')
            axs.grid()

        if plot_boundary:
            # extract upper and lower bounds
            if state_dim > 1:
                for axs_num, bounds in bdry_dict.items():
                    axs[axs_num].axhline(y=bounds[0], color='k', linestyle='--')
                    axs[axs_num].axhline(y=bounds[1], color='k', linestyle='--')
            else:
                bounds = bdry_dict.get(0)  # if the dim is less than 1 then the key has to be for 1st state
                axs.axhline(y=bounds[0], color='k', linestyle='--')
                axs.axhline(y=bounds[1], color='k', linestyle='--')


        # The key option here is `bbox`. I'm just going a bit crazy with it.
        # if state_dim > 1:
        #     axs[ax_id].annotate('{:.{}f}'.format(data_array[0, ax_id], 5), xy=(1, data_array[0, ax_id]),
        #                         xytext=xyloc,
        #                         textcoords='offset points', ha='center', va='bottom',
        #                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        #                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
        #                                         color='red'))
        # else:
        #     axs.annotate('{:.{}f}'.format(data_array[0, ax_id], 5), xy=(1, data_array[0, ax_id]),
        #                  xytext=xyloc,
        #                  textcoords='offset points', ha='center', va='bottom',
        #                  bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        #                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
        #                  color='red'))

    # init_state = state_evolution[0]
    # init_state[2] = init_state[2] * 180 / math.pi
    # init_state[3] = init_state[3] * 180 / math.pi
    # axs[0].text(0.8, 1.2,
    #             "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
    #             horizontalalignment='center', verticalalignment='center',
    #             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
    #             transform=axs[0].transAxes)

    plt.plot()
    # axs[0].set_xticklabels([])
    # axs[1].set_xticklabels([])
    # axs[2].set_xticklabels([])
    # axs[-1].set_xlabel('time-step')
    # fig.align_ylabels(axs)
    # plt.savefig("/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_controller.png", dpi=300)
    if block_plot:
        # plt.show(block=True)
        plt.savefig("/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_controller.png", dpi=300)
    else:
        plt.show()


def simulate_toy_2d_system(system_state: np.array,
                           mat_file_dir: str,
                           rollout_steps: int = 100,
                           num_controllers: int = 2,
                           print_flag: bool = False,
                           use_controller: bool = False,
                           visulaize_traj: bool = True):
    """

    :return: 
    """

    toy_model_dir = "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/crown_toolbox/" \
                    "models/my_model_sys_2"

    # load the file
    control_dict = loadmat(mat_file_dir)
    eps = control_dict.get("eps")
    state_space = control_dict.get("state_space")
    s_dim = state_space.shape[0]  # System dimension

    # get partitions to create the look up dictionary
    partition_from_mat = control_dict.get('partitions')

    # get upper & lowe bound A matrices
    A_ub = control_dict.get('M_h')
    A_lb = control_dict.get('M_l')

    # get upper & lower bound B matrices
    b_ub = control_dict.get('B_h')
    b_lb = control_dict.get('B_l')

    # process it before creating the dict
    processed_partitions = postprocess_partition_dump(partition_from_mat)
    A_ub = postprocess_partition_dump(A_ub)
    A_lb = postprocess_partition_dump(A_lb)

    if num_controllers == 2:
        control_coeff_matrix1 = control_dict.get('control_matrix1')
        control_coeff_matrix2 = control_dict.get('control_matrix2')

        control_coeff_matrix = (control_coeff_matrix1, control_coeff_matrix2)
    else:
        control_coeff_matrix = control_dict.get('control_matrix1')

        # DEBUGGING -04/16 625 hypercubes with new controller
        # 109 : [[-1.7 -1.5], [-0.7, -0.5]]
        # control_coeff_matrix[109] = [0.29020143776351737, 0.09840161892983938, -0.10347471780209735,
        #                              -0.16471135075735918, -0.42579583741951793, 0.733109170425697]

        control_coeff_matrix[109] = [-0.03191552522987342, 0.09248865626223877, -0.07110444775385988,
                                     -0.03352390467665076, 0.042838454246428286, -0.010810181546633863]

        # 210 : [[ -0.9, -0.7],[-0.5, -0.3]]
        control_coeff_matrix[210] = [1.9773979444893444, 0.00987073336413336, 0.7107408495056129,
                                     -0.4649849840928639, -0.010050395846163197, 2.2713231162157475]

    _, partition_dim_list = get_hypercube(eps=eps[0, 0], state_space=state_space, n=s_dim, print_flag=print_flag)

    # create the hypercube lookup dict
    hypercube_idx_dict = create_hypercube_idx_dict(processed_partitions)

    state_evolution = np.zeros(shape=(rollout_steps + 1, s_dim))
    control_evolution = np.zeros(shape=(rollout_steps, num_controllers))
    curr_state = system_state.reshape(1, 2)
    state_evolution[0] = curr_state

    tf_toy_system = tf.keras.models.load_model(toy_model_dir)

    # plot for sanity check
    # plot_learned_sys_phase_portrait(input_data=np.array([[-10, -10], [10, 10]]), trained_nn_model=tf_toy_system)

    for step in range(rollout_steps):
        # get next state
        if use_controller:
            tmp_next_state = tf_toy_system.predict(curr_state)
            # tmp_next_state = curr_state
            next_state = evolve_according_to_controller(partitions=processed_partitions,
                                                        partition_dim_list=partition_dim_list,
                                                        curr_state=curr_state,
                                                        next_state=tmp_next_state,
                                                        A_ub=A_ub,
                                                        A_lb=A_lb,
                                                        b_ub=b_ub,
                                                        b_lb=b_lb,
                                                        hypercube_idx_dict=hypercube_idx_dict,
                                                        control_evolution=control_evolution,
                                                        control_coeff_matrix=control_coeff_matrix,
                                                        num_controllers=num_controllers,
                                                        time_step=step,
                                                        epsilon=eps[0, 0],
                                                        print_flag=False)
            state_evolution[step + 1] = next_state

        else:
            next_state = tf_toy_system.predict(curr_state)
            state_evolution[step + 1] = next_state

        curr_state = next_state.reshape(1, 2)

    if visulaize_traj:
        fig, axs = plt.subplots(1)
        fig.suptitle('2d system trajectory')

        plt.plot(state_evolution[:, 0], state_evolution[:, 1], 'ro', linestyle='-.')
        plt.show(block=True)
    else:
        plot_2d_toy_sys_from_data_file(state_evolution=control_evolution)


if __name__ == "__main__":
    pendulum_control_mat_file = ""

    system_state = np.array([1e-8, 1e-8])  # 2d array
    # system_state = np.arrya([0, 0, 0]) # 3d array

    system_state = np.array([-1.6, -0.6])  # 2d array - 04/16 debugging - Cube 109
    # system_state = np.array([-0.8, -0.4])  # 2d array - 04/16 debugging - Cube 210

    simulate_toy_2d_system(system_state=system_state,
                           mat_file_dir="/home/karan/Documents/research/"
                                        "nn_veri_w_crown/rl_train_agent/"
                                        "pendulum_control_data/control_lags_twodim_625_u_0.5.mat",
                           use_controller=True,
                           rollout_steps=10,
                           num_controllers=1,
                           visulaize_traj=True)
