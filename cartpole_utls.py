import time
import struct
import gym
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

from scipy.io import loadmat
from gen_utls import simulate_zamani_ex_1, create_hypercube_idx_dict, evolve_according_to_controller,\
    postprocess_partition_dump, get_hypercube

from gym.wrappers.monitoring import video_recorder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DIR_PATH = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent"


def simulate_cartpole_w_cbf(system_state: np.array,
                            mat_file_dir: str,
                            rollout: int = 100,
                            num_controllers: int = 2,
                            print_flag: bool = False,
                            use_controller: bool = False,
                            visualize: bool = False,
                            record_flag: bool = False):
    """
    A helper function to simulate the cartpole behavior w cbf
    :return:
    """
    cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_models/" \
                   "cartpole_model_1_cross_128_loss_0.01_data_2500"

    # load the file
    control_dict = loadmat(mat_file_dir)
    eps = control_dict.get("eps")
    state_space = control_dict.get("state_space")
    s_dim = state_space.shape[0]  # System dimension
    if num_controllers == 2:
        control_coeff_matrix1 = control_dict.get('control_matrix1')
        control_coeff_matrix2 = control_dict.get('control_matrix2')

        control_coeff_matrix = (control_coeff_matrix1, control_coeff_matrix2)
    else:
        control_coeff_matrix = control_dict.get('control_matrix1')

    _, partition_dim_list = get_hypercube(eps=eps[0, :], state_space=state_space, n=s_dim, print_flag=print_flag)

    # get partitions to create the look up dictionary
    partition_from_mat = control_dict.get('partitions')

    # process it before creating the dict
    processed_partitions = postprocess_partition_dump(partition_from_mat)

    # create the hypercube lookup dict
    hypercube_idx_dict = create_hypercube_idx_dict(processed_partitions)

    # rollout trajectory
    cartpole = gym.make("CartPole-v0")
    cartpole_model = tf.keras.models.load_model(cartpole_dir)
    cartpole.reset()
    cartpole.render()

    if record_flag:
        vid = video_recorder.VideoRecorder(cartpole,
                                           path='/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/'
                                                'cbf_videos/video_0.mp4')

    # start simulation - keep track of states
    state_evolution = np.zeros(shape=(rollout+1, s_dim))
    control_evolution = np.zeros(shape=(rollout, num_controllers))
    curr_state = system_state.reshape(1, 4)
    state_evolution[0] = curr_state

    for step in range(rollout):
        # get next state
        if use_controller:
            tmp_next_state = cartpole_model.predict(curr_state)
            next_state = evolve_according_to_controller(partitions=processed_partitions,
                                                        partition_dim_list=partition_dim_list,
                                                        curr_state=tmp_next_state,
                                                        hypercube_idx_dict=hypercube_idx_dict,
                                                        control_evolution=control_evolution,
                                                        control_coeff_matrix=control_coeff_matrix,
                                                        num_controllers=num_controllers,
                                                        time_step=step,
                                                        print_flag=False)

            state_evolution[step+1] = next_state

        else:
            next_state = cartpole_model.predict(curr_state)
            state_evolution[step+1] = next_state

        cartpole.unwrapped.state = next_state.reshape(4, )
        if record_flag:
            vid.capture_frame()
        cartpole.render()
        curr_state = next_state.reshape(1, 4)

    cartpole.close()

    if visualize:
        plot_from_data_file(state_evolution=control_evolution)

    ''' Save the model in the data folder '''
    _dump_model = input("Do you want to save the strategy,Enter: Y/y")

    # save learned model
    if _dump_model == "y" or _dump_model == "Y":
        # 0 - year; 1 - month ; 2 -day; 3-hour, 4-min, 5-sec
        time_struct = time.localtime(time.time())
        file_name = f"pendulum_control_data/cartpole_model_5_rollout_{rollout}_{time_struct[1]}_{time_struct[2]}" \
                               f"_{time_struct[3]}_{time_struct[4]}_{time_struct[5]}.npy"

        file_path = DIR_PATH + "/" + file_name
        # dump states using numpy
        with open(file_name, 'wb') as file_handle:
            np.save(file_handle, state_evolution)


def plot_from_data_file(file_path: str = '', state_evolution=None):
    """
    A helper function to plot using the data file.
    :param file_path:
    :return:
    """

    if state_evolution is None:
        with open(file_path, 'rb') as file_handle:
            state_evolution = np.load(file_handle,  allow_pickle=True)

    assert len(state_evolution.shape) == 2, "Make sure the data matrix is a 2d matrix"
    state_dim = state_evolution.shape[1]

    fig, axs = plt.subplots(state_dim)
    fig.suptitle('Cartpole Model - Trajectory in each dimension')

    bdry_mat = np.array([[-1, 1], [-0.5, 0.5], [-5, 5], [-28.647, 28.647]])

    ylabels: list = [r"$x (m)$", r"$\dot{x} (m/s)$", r"$\theta (deg)$", r"$\dot{\theta} (deg/s)$"]
    color = ['tab:red']
    xyloc: tuple = (-20, 20)

    for ax_id in range(state_dim):
        # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        if ax_id == 2 or ax_id == 3:
            axs[ax_id].plot(state_evolution[:, ax_id] * 180/math.pi, color[0], label=ylabels[ax_id])
        else:
            axs[ax_id].plot(state_evolution[:, ax_id], color[0], label=ylabels[ax_id])
        axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
        axs[ax_id].legend(loc='best')
        axs[ax_id].grid()

        # axs[ax_id].axhline(y=bdry_mat[ax_id, 0], color='k', linestyle='--')
        # axs[ax_id].axhline(y=bdry_mat[ax_id, 1], color='k', linestyle='--')

        # The key option here is `bbox`. I'm just going a bit crazy with it.
        axs[ax_id].annotate('{:.{}f}'.format(state_evolution[0, ax_id], 5), xy=(1, state_evolution[0, ax_id]),
                            xytext=xyloc,
                            textcoords='offset points', ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                            color='red'))
    # init_state = state_evolution[0]
    # init_state[2] = init_state[2] * 180 / math.pi
    # init_state[3] = init_state[3] * 180 / math.pi
    # axs[0].text(0.8, 1.2,
    #             "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
    #             horizontalalignment='center', verticalalignment='center',
    #             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
    #             transform=axs[0].transAxes)

    plt.plot()
    plt.show(block=True)


def simulate_cartpole_from_data_file(file_path: str,):
    """
    A helper function to plot using the data file.
    :param file_path:
    :return:
    """

    with open(file_path, 'rb') as file_handle:
        state_evolution = np.load(file_handle,  allow_pickle=True)

    assert len(state_evolution.shape) == 2, "Make sure the data matrix is a 2d matrix"

    rollout = state_evolution.shape[0]
    state_dim = state_evolution.shape[1]

    cartpole = gym.make("CartPole-v0")
    cartpole.reset()

    curr_state = state_evolution[0]
    cartpole.unwrapped.state = curr_state.reshape(state_dim, )
    cartpole.render()

    for step in range(1, rollout):
        cartpole.unwrapped.state = state_evolution[step].reshape(state_dim, )
        cartpole.render()

    cartpole.close()


if __name__ == "__main__":
    # test zamain's example
    test_zamani_example = False
    if test_zamani_example:
        simulate_zamani_ex_1(system_state=21, use_controller=True)
        sys.exit(-1)
    else:
        cartpole_control_mat_file = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
                                    "pendulum_control_data/updated_control_lags_cartpole_960_new.mat"
        # cartpole_control_mat_file = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
        #                             "pendulum_control_data/control_lags_cartpole_960_updated.mat"

        state_list = [np.array([0, 0, -5*math.pi/180, 0]),
                      # np.array([0, 0, -3*math.pi/180, 0]),
                      # np.array([0, 0, -1*math.pi/180, 0]),
                      # np.array([0, 0, 1*math.pi/180, 0]),
                      # np.array([0, 0, 3*math.pi/180, 0]),
                      # np.array([0, 0, 5*math.pi/180, 0])
                      ]
        # _system_state = np.array([0, 0, -5*math.pi/180, 0])

        for _system_state in state_list:
        # simulate cartpole behavior
            simulate_cartpole_w_cbf(system_state=_system_state,
                                    mat_file_dir=cartpole_control_mat_file,
                                    print_flag=True,
                                    use_controller=True,
                                    visualize=True,
                                    rollout=100)

        file_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_control_data/" \
                    "cartpole_model_5_rollout_100_2_22_17_58_47.npy"

        plot_from_data_file(file_path=file_path)

        # simulate_cartpole_from_data_file(file_path=file_path)
