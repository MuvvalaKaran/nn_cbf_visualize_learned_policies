import  matplotlib.pyplot as plt

import numpy as np

from tf_husky_models import get_trained_husky_models, _get_husky_tf_model_3_cross_x_model_3,\
    _get_husky_tf_model_1_cross_x_model_6, _get_husky_tf_model_1_cross_x_model_5,\
    _get_husky_tf_model_1_cross_x_model_8, _get_husky_tf_model_1_cross_x_model_9,\
    _get_husky_tf_model_2_cross_x_model_

def plot_trained_husky_w_no_mean_and_std(save_figs: bool = False):
    """
     A helper function
    :param save_figs:
    :return:
    """

    roll_out_steps = 20
    # init_state = np.array([0.0, 0.0, 0, 0.1])
    init_list = [np.array([0.0, 0.0, 0, 0]),
                 np.array([0.1, 0.1, 0, 0]),
                 np.array([0.1, -0.1, 0, 0]),
                 np.array([-0.1, 0.1, 0, 0]),
                 # np.array([0.25, 0, 0, 0])
                 ]

    # init_list = [init_state]
    state_space_dim = 4

    def _wrap_pi(phase):
        phases = np.arctan2(np.sin(phase), np.cos(phase))
        return phases

    # mu, sigma = 0, 0.1  # mean and standard deviation
    # np.random.seed(1)
    # for _ in range(10):
    #     init_list.append(np.random.normal(mu, sigma, size=(4,)))
    fig, axs = plt.subplots(4)
    fig.suptitle('TF husky models')
    fig.set_size_inches(18.5, 10.5, forward=True)
    for counter, init_state in enumerate(init_list):
        # init_state = np.random.normal(mu, sigma, size=(4,))

        # for model_num in [1, 2]:
        tf_model_model_3 = _get_husky_tf_model_3_cross_x_model_3(hidden_neurons=256)
        tf_model_model_5 = _get_husky_tf_model_1_cross_x_model_5(hidden_neurons=256)
        tf_model_model_6 = _get_husky_tf_model_1_cross_x_model_6(hidden_neurons=256)
        tf_model_model_8 = _get_husky_tf_model_1_cross_x_model_8(hidden_neurons=256)
        tf_model_model_9 = _get_husky_tf_model_1_cross_x_model_9(hidden_neurons=256)
        tf_2_layer_model_1 = _get_husky_tf_model_2_cross_x_model_(hidden_neurons=256,
                                                                  husky_dir_path="/home/karan/Documents/research/" \
                                                                                 "nn_veri_w_crown/rl_train_agent/" \
                     "husky_2_layer_models/husky_256_256_1/variables/variables")
        tf_2_layer_model_2 = _get_husky_tf_model_2_cross_x_model_(hidden_neurons=256,
                                                                  husky_dir_path="/home/karan/Documents/research/" \
                                                                                 "nn_veri_w_crown/rl_train_agent/" \
                                                                                 "husky_2_layer_models/"
                                                                                 "husky_256_256_2/variables/variables")

        _states_model_3 = np.zeros(shape=(roll_out_steps, state_space_dim))
        _states_model_5 = np.zeros(shape=(roll_out_steps, state_space_dim))
        _states_model_6 = np.zeros(shape=(roll_out_steps, state_space_dim))
        _states_model_8 = np.zeros(shape=(roll_out_steps, state_space_dim))
        _states_model_9 = np.zeros(shape=(roll_out_steps, state_space_dim))

        _curr_state = init_state
        _states_model_3[0] = _curr_state
        _states_model_5[0] = _curr_state
        _states_model_6[0] = _curr_state
        _states_model_8[0] = _curr_state
        _states_model_9[0] = _curr_state

        for step in range(roll_out_steps-1):
            state_pre = tf_model_model_3.predict(_states_model_3[step].reshape(1, 4))
            state_pre = _states_model_3[step] + state_pre
            _states_model_3[step + 1] = state_pre

            state_pre = tf_model_model_5.predict(_states_model_5[step].reshape(1, 4))
            state_pre = _states_model_5[step] + state_pre
            _states_model_5[step + 1] = state_pre

            state_pre = tf_model_model_6.predict(_states_model_6[step].reshape(1, 4))
            state_pre = _states_model_6[step] + state_pre
            _states_model_6[step + 1] = state_pre

            state_pre = tf_model_model_8.predict(_states_model_8[step].reshape(1, 4))
            state_pre = _states_model_8[step] + state_pre
            _states_model_8[step + 1] = state_pre

            state_pre = tf_model_model_9.predict(_states_model_9[step].reshape(1, 4))
            state_pre = _states_model_9[step] + state_pre
            _states_model_9[step + 1] = state_pre

            # state_pre = np.round(state_pre, decimals=2)

            # pass theta and wrap if needed
            # new_theta = _wrap_pi(state_pre[0, 2])
            # if not np.isclose(new_theta, state_pre[0, 2], atol=0.1):
            #     print("Before theta val: {}".format(state_pre[0, 2]))
            #     print("After theta val: {}".format(new_theta))
            #     state_pre[0, 2] = new_theta
            # _states_model_3[step + 1] = state_pre

        color = ['tab:red', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:purple']
        legend = "Model 5"
        ylabels: list = [r"$x$", r"$y$", r"$\theta$", r"$v$"]

        ### plot the behavior
        # for ax_id in range(state_space_dim):
        #     # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        #     axs[ax_id].plot(_states_model_3[:, ax_id], color[0], label="Model 3")
        #     axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
        #     axs[ax_id].legend(loc='best')
        #
        #     # The key option here is `bbox`. I'm just going a bit crazy with it.
        #     axs[ax_id].annotate('{:.{}f}'.format(_states_model_3[0, ax_id], 5), xy=(1, _states_model_3[0, ax_id]),
        #                         xytext=(-20, 20),
        #                         textcoords='offset points', ha='center', va='bottom',
        #                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        #                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
        #                                         color='red'))

        ### plot the behavior
        # for ax_id in range(state_space_dim):
        #     # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        #     axs[ax_id].plot(_states_model_5[:, ax_id], color[1], label="Model 5")
        #     axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
        #     axs[ax_id].legend(loc='best')
        #
        #     # The key option here is `bbox`. I'm just going a bit crazy with it.
        #     axs[ax_id].annotate('{:.{}f}'.format(_states_model_5[0, ax_id], 5), xy=(1, _states_model_5[0, ax_id]),
        #                         xytext=(-20, 20),
        #                         textcoords='offset points', ha='center', va='bottom',
        #                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        #                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
        #                                         color='red'))

        ### plot the behavior
        # for ax_id in range(state_space_dim):
        #     # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        #     axs[ax_id].plot(_states_model_6[:, ax_id], color[2], label="Model 6")
        #     axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
        #     axs[ax_id].legend(loc='best')
        #
        #     # The key option here is `bbox`. I'm just going a bit crazy with it.
        #     axs[ax_id].annotate('{:.{}f}'.format(_states_model_6[0, ax_id], 5), xy=(1, _states_model_6[0, ax_id]),
        #                         xytext=(-20, 20),
        #                         textcoords='offset points', ha='center', va='bottom',
        #                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        #                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
        #                                         color='red'))

        ### plot the behavior
        for ax_id in range(state_space_dim):
            # axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
            axs[ax_id].plot(_states_model_8[:, ax_id], color[2], label="Model 8")
            axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
            axs[ax_id].legend(loc='best')

            # The key option here is `bbox`. I'm just going a bit crazy with it.
            axs[ax_id].annotate('{:.{}f}'.format(_states_model_8[0, ax_id], 5), xy=(1, _states_model_8[0, ax_id]),
                                xytext=(-20, 20),
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                color='red'))

        ### plot the behavior
        for ax_id in range(state_space_dim):
            # axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
            axs[ax_id].plot(_states_model_8[:, ax_id], color[2], label="Model 8")
            axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
            axs[ax_id].legend(loc='best')

            # The key option here is `bbox`. I'm just going a bit crazy with it.
            axs[ax_id].annotate('{:.{}f}'.format(_states_model_8[0, ax_id], 5), xy=(1, _states_model_8[0, ax_id]),
                                xytext=(-20, 20),
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                color='red'))

        plt.plot()

    axs[0].text(0.8, 1.2,
                "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                transform=axs[0].transAxes)

    if save_figs:
        fig_dir_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/plots"

        fig_name = 'tf_husky_{}_x_{}_y_{}_theta_{}_v_{}.png'.format(model_num,
                                                                    init_state[0],
                                                                    init_state[1],
                                                                    init_state[2],
                                                                    init_state[3])
        full_name = fig_dir_path + fig_name
        fig.savefig(full_name, dpi=100)

    plt.show(block=True)


def plot_trained_husky_w_mean_and_std(save_figs: bool = True):
    """
     A helper function
    :param save_figs:
    :return:
    """
    init_state = np.array([0.5, 0.5, 0, 0])
    # init_list = [np.array([0.0, 0.0, 0, 0]),
    #              np.array([0.1, 0.1, 0, 0]),
    #              np.array([0.1, -0.1, 0, 0]),
    #              np.array([-0.1, 0.1, 0, 0]), np.array([0.25, 0, 0, 0])]

    init_list = [init_state]

    # mu, sigma = 0, 0.1  # mean and standard deviation
    # np.random.seed(1)
    # for _ in range(10):
    #     init_list.append(np.random.normal(mu, sigma, size=(4,)))

    for init_state in init_list:


        # init_state = np.random.normal(mu, sigma, size=(4,))

        fig, axs = plt.subplots(4)
        fig.suptitle('TF husky models')
        fig.set_size_inches(18.5, 10.5, forward=True)

        for model_num in [1, 2]:
            tf_model, ip_weights, op_weights = get_trained_husky_models(which_model=model_num)

            _mean_ip = ip_weights[0]
            _std_ip = ip_weights[1]
            _mean_op = op_weights[0]
            _std_op = op_weights[1]

            state_space_dim = 4

            _states = np.zeros(shape=(500, state_space_dim))
            _curr_state = init_state
            _states[0] = _curr_state
            for step in range(499):
                curr_state_preprocessed = _states[step] - _mean_ip
                curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed / _std_ip)

                state_pre = tf_model.predict(curr_state_preprocessed.reshape(1, 4))

                processed_state = (state_pre * _std_op) + _mean_op
                _states[step + 1] = processed_state

            color = ['tab:red', 'tab:green']
            legend = "norm {}".format(model_num)
            ylabels: list = [r"$x$", r"$y$", r"$\theta$", r"$v$"]

            ### plot the behavior
            for ax_id in range(state_space_dim):
                # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
                axs[ax_id].plot(_states[:, ax_id], color[model_num-1], label=legend)
                axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
                axs[ax_id].legend(loc='best')

                # The key option here is `bbox`. I'm just going a bit crazy with it.
                axs[ax_id].annotate('{:.{}f}'.format(_states[0, ax_id], 5), xy=(1, _states[0, ax_id]),
                                    xytext=(-20, 20),
                                    textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                    color='red'))

        axs[0].text(0.8, 1.2,
                    "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    transform=axs[0].transAxes)

        if save_figs:
            fig_dir_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/plots"

            fig_name = 'tf_husky_{}_x_{}_y_{}_theta_{}_v_{}.png'.format(model_num,
                                                                        init_state[0],
                                                                        init_state[1],
                                                                        init_state[2],
                                                                        init_state[3])
            full_name = fig_dir_path + fig_name
            fig.savefig(full_name, dpi=100)

    plt.show(block=True)
