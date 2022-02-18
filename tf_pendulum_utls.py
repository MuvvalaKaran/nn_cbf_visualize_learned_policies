import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from tf_pendulum_models import get_pendulum_tf_model_1_cross_x_models
from gym.wrappers.monitoring import video_recorder


def plot_delta_models_fixed_init_state():
    deg = 0
    thdot = 0

    # fig, axs = plt.subplots(3)
    # fig.suptitle(f'Pendulum Stabilization -model deg {deg}')
    # fig.set_size_inches(18.5, 10.5, forward=True)

    color_scheme: list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:brown', 'tab:cyan', 'tab:purple',
                          'tab:olive', 'tab:pink', 'tab:gray']
    compare_pendulum_models_stability(use_multiple_models=True,
                                      use_akashs_model=False,
                                      # init_state=np.array([math.cos(deg * 0.0175), math.sin(deg * 0.0175),
                                      #                      thdot]),
                                      init_state=np.array([0, 0,
                                                           thdot]),
                                      color=color_scheme[0],
                                      legend=f'{thdot} thdot',
                                      fig_axs=None)

    plt.show(block=True)


def plot_delta_model_vs_deg_vs_thdot(save_flag: bool = False):
    """
    A helper function to a model's performance for various theta vs thetadot

    :return:
    """
    degs = np.arange(-20, 21, 5)
    theta_dot = np.arange(-2.5, 2.5, 0.5)
    color_scheme: list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:brown', 'tab:cyan', 'tab:purple',
                          'tab:olive', 'tab:pink', 'tab:gray']
    # legend_list = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']

    for c_counter, deg in enumerate(degs):
        # fig, axs = plt.subplots(3)
        # fig.suptitle(f'Pendulum Stabilization -model deg {deg}')
        # fig.set_size_inches(18.5, 10.5, forward=True)

        for l_counter, thdot in enumerate(theta_dot):
            compare_pendulum_models_stability(use_akashs_model=False,
                                              model_dir_path=
                                              "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",
                                              init_state=np.array([math.cos(deg * 0.0175), math.sin(deg * 0.0175),
                                                                   thdot]),
                                              color=color_scheme[l_counter],
                                              legend=f'{thdot}',
                                              fig_axs=None)
            plt.plot()

        if save_flag:
            fig_dir_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_models/"
            fig_name = f'pendulum_model_evolution_deg_{deg}.png'
            full_name = fig_dir_path + fig_name
            fig.savefig(full_name)
        else:
            plt.show(block=True)

    # if not save_flag:
    #     plt.show(block=True)


def animate_nn_behavior(state_matrix):
    # ANIMATION FUNCTION
    def func(num, dataSet, line, redDots):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        redDots.set_data(dataSet[0:2, :num])
        redDots.set_3d_properties(dataSet[2, :num])
        return line

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    fig, ax = plt.subplots()
    # fig = plt.figure(1)
    # ax = Axes3D(fig)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    # ax.set_zlabel(r"$\dot{\theta}$")
    plt.title("System Behavior")

    # line = plt.plot(state_matrix[:, 0], state_matrix[:, 1], state_matrix[:, 2], lw=2, c='g')[0]
    numDataPoints = state_matrix.shape[0]
    for num in range(numDataPoints):
        # redDots = plt.plot(state_matrix[num, 0], state_matrix[num, 1], state_matrix[num, 2], lw=2, c='r',  marker='o')[0]
        redDots = plt.plot(state_matrix[num, 0], state_matrix[num, 1], lw=2, c='g', marker='o')
        plt.pause(0.1)

    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(np.array(state_matrix), line, redDots),
                                       interval=5000, blit=False)
    # ax.plot3D(state_matrix[:, 0], state_matrix[:, 1], state_matrix[:, 2])
    plt.show()
    # plt.show(block=True)
    sys.exit(-1)





def plot_3d_data():
    data_dir_list = [
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_2500.txt",
        # "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_5000.txt",
        # "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_7500.txt",
        # "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_10000.txt",
    ]

    for data_dir in data_dir_list:
        data = pd.read_csv(data_dir)

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        # IP
        X_data = data[["ip1", "ip2", "ip3"]].values

        # OP
        Y_data = data[["op1", "op2", "op3"]].values

        X = X_data[:, 0][:, np.newaxis].T
        Y = X_data[:, 1][:, np.newaxis].T
        Z = X_data[:, 2][:, np.newaxis].T
        # C = X_data[:, 3][:, np.newaxis].T

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\dot{\theta}$")
        plt.title("Input Data visualization - Data:{}".format(X_data.shape[0]))

        img = ax.scatter(X, Y, Z)#, c=C, cmap=plt.get_cmap('Blues'))

        X = Y_data[:, 0][:, np.newaxis].T
        Y = Y_data[:, 1][:, np.newaxis].T
        Z = Y_data[:, 2][:, np.newaxis].T
        # C = X_data[:, 3][:, np.newaxis].T

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\dot{\theta}$")
        plt.title("Output Data visualization - Data:{}".format(Y_data.shape[0]))

        # fig.colorbar(img)
    plt.show()


def get_angle_from_x_y_coordinates(x_coord, y_coord):
    # no clipping on theta in gym - only for computing costs.
    theta = math.atan2(y_coord, x_coord)  # theta is in radians
    return theta


def get_3d_coords_from_2_coords(_2d_state):
    """
    A helper function to convert 2d state used by the pendulum gym render() function to 3d coordinate used NN learned
    system

    :param _2d_state:
    :return:
    """
    x_coord, y_coord = get_x_y_from_theta(_2d_state[0])

    return np.array([x_coord, y_coord, _2d_state[1]])



def get_x_y_from_theta(theta_rad):
    x = math.cos(theta_rad)
    y = math.sin(theta_rad)

    return x, y


def simulate_learned_pendulum_model(model,
                                    gym_env,
                                    theta = None,
                                    init_state = None,
                                    record_flag: bool = False,
                                    rollout_steps: int = 500):
    if theta is None:
        if isinstance(init_state, type(None)):
            theta = np.random.uniform(low=-math.pi/180, high=math.pi/180)
            theta = 15 * -math.pi/180
            x = math.cos(theta)
            y = math.sin(theta)
            init_state = np.array([0, 1, 0])

            ## test
            theta_test = get_angle_from_x_y_coordinates(x_coord=x, y_coord=y)
            _2d_init_state = np.array([theta_test, 0])

            print(f"Stating the pendulum at theta: {theta_test * 180/math.pi} degrees")
    else:
        print(f"Stating the pendulum at theta: {theta * 180/math.pi} degrees")
        _2d_init_state = np.array([theta, 0])

    gym_env.reset()
    gym_env.unwrapped.state = _2d_init_state
    gym_env.render()

    if record_flag:
        vid = video_recorder.VideoRecorder(gym_env, path='/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/NN_videos/video_5.mp4')

    # start simulation
    curr_state = _2d_init_state.reshape(2, )
    for steps in range(rollout_steps):
        # get next state
        _3d_state = get_3d_coords_from_2_coords(curr_state)
        next_state = model.predict(_3d_state.reshape(1, 3))
        next_state = np.array([get_angle_from_x_y_coordinates(x_coord=next_state[0, 0], y_coord=next_state[0, 1]),
                               next_state[0, 2]])

        gym_env.unwrapped.state = next_state.reshape(2, )
        if record_flag:
          vid.capture_frame()
        gym_env.render()
        curr_state = next_state

    gym_env.close()


def simulate_pendulum(gym_env):
    
    """model 1 falls in +region
    model 2 - -ve just stay still
    +ve very bad
    model 3 - extrem good
    model 4 - good
    model 5 - reversedirectopn
    model 6 - very bad extreme
    model7 - very bad extreme
    model8 - very bad extreme

    Akash model 128 - 1- bad in extreme - but moves a little after settling down
    Akash model 128 - 2- bad in extreme - but moves a little after settling down
    Akash model 64 -2 - same behavior
    Akash model 64 -1 - has a nice behavior where it oscillates at the bottom after falling - more natural motion than
                        other models
    """

    model_dirs = [
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_2500",  # model 1
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_5000", # model 2
    "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",  # model 3
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_7500",  # model 4
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.001_data_7500",  # model 5
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_10000",  # model 6
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.01_data_2500",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.001_data_2500",  # model 7
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.0015_data_5000",  # model 8
    ]

    model_dirs = ['1, ']

    for counter, model_path in enumerate(model_dirs):
        # load model weights
        model = tf.keras.models.load_model(filepath=model_path)
        #
        # model = get_pendulum_tf_model_1_cross_x_models(hidden_neurons=64,
        #                                                which_model=1)

        # loop for theta
        thetas = np.arange(-45, 45, 5)
        print("********************************************************************************************")
        print("Testing Model {}".format(counter))
        print("********************************************************************************************")
        for theta in thetas:
            simulate_learned_pendulum_model(model,
                                            gym_env,
                                            theta=theta*math.pi/180,
                                            # init_state=np.array([1/sqrt(2), 1/sqrt(2), 0.0]),
                                            # init_state=np.array([0.7, 0.7, 0.0]),
                                            rollout_steps=200)


def _plot_pendulum_state_evolution(dir_path,
                                   use_akash_model_flag,
                                   init_state,
                                   fig_axs,
                                   color,
                                   legend,
                                   xyloc: tuple = (-20, 20),
                                   rollout: int = 100):
    """
    A helper function to plot the evolution of change in each state in the pendulum model
    :return:
    """
    if not use_akash_model_flag:
        model = tf.keras.models.load_model(dir_path)
    else:
        # _hidden_neurons = input(" \n Enter # of hidden neurons:")
        # _model_num = input(" \n Enter Model num:")
        _hidden_neurons = 128
        _model_num = 2
        model = get_pendulum_tf_model_1_cross_x_models(hidden_neurons=int(_hidden_neurons),
                                                       which_model=int(_model_num))

    observation = init_state
    # col = stat diff # row = time step
    state_diff = np.zeros(shape=(rollout, observation.shape[1]))
    for step in range(rollout):
        # roll out
        next_state = model.predict(observation)
        # next_state = akash_model.predict(observation)
        # state_diff[step] = abs(next_state - observation)
        state_diff[step] = next_state
        observation = next_state

    # add function to plot in matplotlib
    animate_nn_behavior(state_diff)

    # ylabels: list = [r"$\Delta x$", r"$\Delta y$", r"$\Delta \dot{\theta}$"]
    ylabels: list = [r"$x$", r"$y$", r"$\dot{\theta}$"]

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

    # plt.plot()


def compare_pendulum_models_stability(fig_axs,
                                      color,
                                      legend,
                                      init_state=None,
                                      use_akashs_model: bool = False,
                                      model_dir_path: str = '',
                                      use_multiple_models: bool = False):
    state_space_dim = 3
    if isinstance(init_state, type(None)):
        mu, sigma = 0, 0.1  # mean and standard deviation
        np.random.seed(2)
        init_state = np.random.normal(mu, sigma, size=(state_space_dim,))

    if use_multiple_models:
        model_dirs = [
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_2500",
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_5000", # - only good in one region of thetadot
            "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_7500", # - kinda blows up at -1.0 but overall performance is good
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_10000", - blows up
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.01_data_2500", # - kinda blows up at 1.0
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.001_data_2500", # blows up at 1.0
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.0015_data_5000", #- only good in one region of thetadot
        ]
    else:
        if model_dir_path is '':
            print("***********************************************************************************")
            print("TELL ME THE MODEL TO PLOT FOR!!!!!")
            print("***********************************************************************************")
            sys.exit(-1)

        model_dirs = [model_dir_path]

    xytests = [(-20, 20), (-20, 30), (-20, 40), (-20, 50), (-20, 60), (-20, 70), (-20, 80)]

    for counter, _model in enumerate(model_dirs):
        _plot_pendulum_state_evolution(dir_path=_model,
                                       use_akash_model_flag=use_akashs_model,
                                       init_state=init_state.reshape(1, state_space_dim),
                                       fig_axs=fig_axs,
                                       color=color,
                                       legend=legend,
                                       xyloc=xytests[counter],
                                       rollout=200)

    fig_axs[0].text(0.8, 1.2,
                    "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    transform=fig_axs[0].transAxes)

    # plt.show(block=True)


