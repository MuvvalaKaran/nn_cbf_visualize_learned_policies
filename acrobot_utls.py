import gym
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tf_toy_system_utls import plot_2d_toy_sys_from_data_file
from tf_acrobot_models import get_acrobot_tf_model_1_cross_x, get_acrobot_tf_model_2_cross_x,\
    get_acrobot_tf_model_3_cross_x

from gen_utls import create_hypercube_idx_dict, evolve_according_to_controller, postprocess_partition_dump,\
    get_hypercube


def plot_acrobot_state_evolution(dir_path,
                                 init_state,
                                 fig_axs, color, legend,
                                 xyloc: tuple = (-20, 20), rollout: int = 100):
    """
    A helper function to plot the evolution to chnage in each state in the cartpole model
    :return:
    """
    # for 1 layer models
    acrobot_model = get_acrobot_tf_model_1_cross_x(hidden_neurons=512,
                                                   variables_path=dir_path,
                                                   print_flag=True)

    # for 2+ layer models
    # acrobot_model = get_acrobot_tf_model_2_cross_x(hidden_neurons=512,
    #                                                variables_path=dir_path,
    #                                                print_flag=True)

    observation = init_state
    # col = stat diff # row = time step
    state_diff = np.zeros(shape=(rollout, observation.shape[1]))
    for step in range(rollout):
        # roll out for 1 layer
        next_state = acrobot_model.predict(observation)

        # for 2+ layers that are trained over drift
        # delta = acrobot_model.predict(observation)
        # next_state = delta + observation
        # state_diff[step] = abs(next_state - observation)
        state_diff[step] = next_state
        observation = next_state

    ylabels: list = [r"$Cos(\theta_1)$", r"$Sin(\theta_1)$", r"$Cos(\theta_1)$", r"$Sin(\theta_1)$", r"$v_1$", r"$v_2$"]

    for ax_id in range(state_diff.shape[1]):
        # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        fig_axs[ax_id].plot(state_diff[:, ax_id], color, label=legend)
        fig_axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
        fig_axs[ax_id].legend(loc='best')

        # The key option here is `bbox`. I'm just going a bit crazy with it.
        fig_axs[ax_id].annotate('{:.{}f}'.format(state_diff[0, ax_id], 5), xy=(1, state_diff[0, ax_id]),
                                xytext=xyloc,
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                color='red'))

    plt.plot()


def simulate_acrobot_w_cbf(system_state: np.array,
                           mat_file_dir: str,
                           rollout: int = 100,
                           print_flag: bool = False,
                           use_controller: bool = False,
                           visualize: bool = False):
    # load the file
    control_dict = loadmat(mat_file_dir)
    eps = control_dict.get("eps")
    state_space = control_dict.get("state_space")
    s_dim = state_space.shape[0]  # System dimension

    control_coeff_matrix = control_dict.get('array_control')

    _, partition_dim_list = get_hypercube(eps=eps[0, :], state_space=state_space, n=s_dim, print_flag=print_flag)

    # get partitions to create the look up dictionary
    partition_from_mat = control_dict.get('partitions')

    # process it before creating the dict
    processed_partitions = postprocess_partition_dump(partition_from_mat)

    # create the hypercube lookup dict
    hypercube_idx_dict = create_hypercube_idx_dict(processed_partitions)

    model_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
                "acrobot_models/1_layer/acrobot_model_1_cross_512_loss_0.001_data_25000/variables/variables"

    # 1 layer Acrobot models
    acrobot_model = get_acrobot_tf_model_1_cross_x(hidden_neurons=512,
                                                   variables_path=model_dir,
                                                   print_flag=True)

    # rollout trajectory
    s_dim = 6
    state_evolution = np.zeros(shape=(rollout + 1, s_dim))
    control_evolution = np.zeros(shape=(rollout, 1))
    curr_state = system_state.reshape(1, s_dim)
    state_evolution[0] = curr_state

    for step in range(rollout):
        # get next state
        if use_controller:
            tmp_next_state = acrobot_model.predict(curr_state)

            next_state = evolve_according_to_controller(partitions=processed_partitions,
                                                        partition_dim_list=partition_dim_list,
                                                        curr_state=curr_state,
                                                        next_state=tmp_next_state,
                                                        A_ub=None,
                                                        A_lb=None,
                                                        b_ub=None,
                                                        b_lb=None,
                                                        hypercube_idx_dict=hypercube_idx_dict,
                                                        control_evolution=control_evolution,
                                                        control_coeff_matrix=control_coeff_matrix,
                                                        num_controllers=1,
                                                        time_step=step,
                                                        epsilon=eps[0, 0],
                                                        print_flag=False,
                                                        use_acrobot=True)

        else:
            # next_state = cartpole_model.predict(curr_state)
            # for 1 layer husky that are trained over drift
            delta = acrobot_model.predict(curr_state)
            # next_state = delta + curr_state
            next_state = delta

            # for 2 layer husky model - they predict next state
            # next_state = husky_model.predict(curr_state)

        state_evolution[step+1] = next_state

        curr_state = next_state.reshape(1, s_dim)

    if visualize:
        plot_2d_toy_sys_from_data_file(data_array=state_evolution,
                                       ylabels=[r"$\cos(\theta_1)$", r"$\sin(\theta_1)$", r"$\cos(\theta_2)$",
                                                r"$\sin(\theta_2)$", r"$\dot{\theta_1}$", r"$\dot{\theta_2}$"],
                                       # ylabels=[r"$u1 $", "$u2 $"],
                                       # title='Acrobot Model - Trajectory in each dimension',
                                       plot_boundary=True,
                                       bdry_dict=dict({0: [-0.1, 0.1],
                                                       2: [-0.1, 0.1],
                                                       1: [-0.6, 0.6],
                                                       3: [-0.6, 0.6],
                                                       4: [-0.25, 0.25],
                                                       5: [-0.25, 0.25]})
                                       )
    


if __name__ == "__main__":
    acrobot_control_mat_file = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
                             "acrobot_control_data/acrobot_partition_data_144.mat"
    
    acrobot = gym.make("Acrobot-v1")
    # lows = [-0.5, -0.5, -0.5, -0.5, -0.1, -0.1]
    # highs = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1]
    # init_state = acrobot.np_random.uniform(low=lows, high=highs, size=(6,))
    state_list = [
        np.array([-0.01, -0.5, 0.0, 0.1, 0.1, 0.1])]

    simulate_acrobot_w_cbf(system_state=state_list[0],
                           mat_file_dir=acrobot_control_mat_file,
                           rollout=20,
                           use_controller=True,
                           visualize=True)