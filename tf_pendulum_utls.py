import sys
import gym
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

from scipy.io import loadmat
from gen_utls import create_hypercube_idx_dict, evolve_according_to_controller, \
    postprocess_partition_dump, get_hypercube, plot_learned_sys_phase_portrait
from tf_pendulum_models import get_pendulum_tf_model_1_cross_x_models, get_2d_pendulum_tf_model_3_cross_x_models
from gym.wrappers.monitoring import video_recorder
from tf_toy_system_utls import plot_2d_toy_sys_from_data_file


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class PendulumAgent:
    def decide(self, observation):
        x, y, angle_velocity = observation
        flip = (y < 0.)
        if flip:
            y *= -1. # now y >= 0
            angle_velocity *= -1.
        angle = np.arcsin(y)
        if x < 0.:
            angle = np.pi - angle
        if (angle < -0.3 * angle_velocity) or \
                (angle > 0.03 * (angle_velocity - 2.5) ** 2. + 1.
                 and angle < 0.15 * (angle_velocity + 3.) ** 2. + 2.):
            force = 2.
        else:
            force = -2.
        if flip:
            force *= -1.
        action = np.array([force,])
        # action = force
        return action


def plot_delta_models_fixed_init_state(system_dimension: int, pendulum_radius=0.5):
    deg = 0
    thdot = 0

    assert pendulum_radius >= 0.1 and pendulum_radius <= 1.0, "Please enter arm length between 0.1 and 1.0"

    fig, axs = plt.subplots(system_dimension)
    fig.suptitle(f'Pendulum Stabilization -model deg {deg}')
    fig.set_size_inches(18.5, 10.5, forward=True)

    if system_dimension == 2:
        init_state = np.array([deg * 0.0175, thdot])
    elif system_dimension == 3:
        init_state = np.array([deg * 0.0175, thdot, pendulum_radius])
    else:
        print("Invalid system dimension length for Pendulum System")
        sys.exit(-1)

    color_scheme: list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:brown', 'tab:cyan', 'tab:purple',
                          'tab:olive', 'tab:pink', 'tab:gray']
    compare_pendulum_models_stability(use_multiple_models=False,
                                      system_dimension=system_dimension,
                                      use_akashs_model=False,
                                      model_dir_path="/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.5_data_7500",
                                      init_state=init_state,
                                      color=color_scheme[0],
                                      legend=f'{thdot} thdot',
                                      fig_axs=axs)

    plt.show(block=True)


def plot_delta_model_vs_deg_vs_thdot(system_dimension: int,
                                     pendulum_radius=0.5,
                                     save_flag: bool = False):
    """
    A helper function to a model's performance for various theta vs thetadot

    :return:
    """
    degs = np.arange(-20, 21, 5)
    theta_dot = np.arange(-0.5, 0.5, 0.1)
    color_scheme: list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:brown', 'tab:cyan', 'tab:purple',
                          'tab:olive', 'tab:pink', 'tab:gray']
    # legend_list = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']

    assert pendulum_radius >= 0.1 and pendulum_radius <= 1.0, "Please enter arm length between 0.1 and 1.0"

    for c_counter, deg in enumerate(degs):
        fig, axs = plt.subplots(system_dimension)
        fig.suptitle(f'Pendulum Stabilization -model deg {deg}')
        fig.set_size_inches(18.5, 10.5, forward=True)

        for l_counter, thdot in enumerate(theta_dot):

            if system_dimension == 2:
                init_state = np.array([deg * 0.0175, thdot])
            elif system_dimension == 3:
                init_state = np.array([deg * 0.0175, thdot, pendulum_radius])
            else:
                print("Invalid system dimension length for Pendulum System")
                sys.exit(-1)

            compare_pendulum_models_stability(use_akashs_model=False,
                                              system_dimension=system_dimension,
                                              use_multiple_models=True,
                                              model_dir_path=
                                              # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",
                                              # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.004_data_7500",
                                              "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.5_data_7500",
                                              # init_state=np.array([math.cos(deg * 0.0175), math.sin(deg * 0.0175),
                                              #                      thdot]),
                                              init_state=init_state,
                                              color=color_scheme[l_counter],
                                              legend=f'{thdot}',
                                              fig_axs=axs)
            plt.plot()

        if save_flag:
            fig_dir_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/new_pendulum_imitation_models/"
            fig_name = f'pendulum_2d_best_model_evolution_deg_{deg}.png'
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


def plot_2d_pendulum_data(scatter_plot: bool = False,
                          scale: float = 1.0):
    data_dir_list = [
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_2500.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_5000.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_7500.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_10000.txt",
    ]

    data = pd.read_csv(data_dir_list[2])

    fig, ax = plt.subplots()
    fig.suptitle(f'Pendulum Input & Output Data')
    fig.set_size_inches(18.5, 10.5, forward=True)

    #IP
    X_data = data[["ip1", "ip2"]].values
    # OP
    Y_data = data[["op1", "op2"]].values

    if scatter_plot:
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\dot{\theta}$")
        ax.scatter(X_data[:, 0], X_data[:, 1])

        ax.scatter(Y_data[:, 0], Y_data[:, 1], c='r')

    else:
        for num in range(X_data.shape[0]):
            curr_state = X_data[num, :]
            op = Y_data[num, :]
            ax.arrow(x=curr_state[0] * 180 / math.pi,
                     y=curr_state[1],
                     dx=scale * (op[0] - curr_state[0]) * 180 / math.pi,
                     dy=scale * (op[1] - curr_state[1]),
                     color='red',
                     head_width=0.05,
                     # length_includes_head=True,
                     alpha=0.5)

    plt.xticks(np.arange(-180, 181, step=10))  # Set label locations.
    plt.grid(visible=True)
    plt.show(block=True)


def simulate_new_3d_pendulum_behavior(init_state: np.array, rollout: int = 100):
    model_dir_list = [
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.01_data_5000",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.01_data_10000",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.5_data_7500"
        ]

    for model_dir in model_dir_list:
        # load model
        model = tf.keras.models.load_model(model_dir)

        observation = init_state.reshape(1, 3)
        # col = stat diff # row = time step
        state_diff = np.zeros(shape=(rollout, observation.shape[1]))
        counter = 0
        for step in range(rollout):
            # roll out
            next_state = model.predict(observation)
            # next_state = akash_model.predict(observation)
            # state_diff[step] = abs(next_state - observation)
            state_diff[step] = next_state
            if next_state[0, 2] <= 0.1 or next_state[0, 2] >= 2.0:
                counter = step
                break
            observation = next_state

        fig, ax = plt.subplots()
        ax.set_ylabel(r"$\theta$")
        ax.set_xlabel(r"$r$")
        # ax.set_zlabel(r"$\dot{\theta}$")
        plt.title("System Behavior")

        # plot the line first and scatter plot manually
        ax.plot(
            state_diff[:counter, 2],
                state_diff[:counter, 0], 'C3', lw=3)
        plt.show()


        numDataPoints = counter
        for num in range(numDataPoints):
            ax.scatter(state_diff[num, 2],
                       state_diff[num, 0],
                       # lw=2,
                       s=120,
                       zorder=2.5,)
                       # linestyle='dashed')
                       # cmap=plt.get_cmap('Blues'),)
                       # marker='o')

        # ax.plot(state_diff[:, 0],
        #         state_diff[:, 1],
        #         # lw=2,
        #         'o-', )
        # linestyle='dashed')
        # cmap=plt.get_cmap('Blues'),)
        # marker='o')
            plt.plot()
            plt.pause(0.1)

        plt.show()







def plot_3d_data():
    data_dir_list = [
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_3d_2500.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_3d_5000.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_3d_7500.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_3d_10000.txt"
        ]

    # data_dir_list = [
    #     "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_2500.txt",
    #     # "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_5000.txt",
    #     # "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_7500.txt",
    #     # "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/pendulum_v0_10000.txt",
    # ]

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

        ax.set_xlabel(r"$\theta_k$")
        ax.set_ylabel(r"$\dot{\theta_k}$")
        ax.set_zlabel(r"$r_k$")
        plt.title("Input Data visualization - Data:{}".format(X_data.shape[0]))

        # img = ax.scatter(X, Y, Z)#, c=C, cmap=plt.get_cmap('Blues'))

        X = Y_data[:, 0][:, np.newaxis].T
        Y = Y_data[:, 1][:, np.newaxis].T
        Z = Y_data[:, 2][:, np.newaxis].T
        # C = X_data[:, 3][:, np.newaxis].T

        ax.set_xlabel(r"$\theta_{k+1}$")
        ax.set_ylabel(r"$\dot{\theta}_{k+1}$")
        ax.set_zlabel(r"$r_{k+1}$")
        # plt.title("Output Data visualization - Data:{}".format(Y_data.shape[0]))

        img = ax.scatter(X, Y, Z)  # , c=C, cmap=plt.get_cmap('Blues'))

        # fig.colorbar(img)
    plt.show()


def get_angle_from_x_y_coordinates(x_coord, y_coord):
    # no clipping on theta in gym - only for computing costs.
    theta = math.atan2(y_coord, x_coord)  # theta is in radians
    return theta


def get_3d_coords_from_2_coords(_2d_state, radius: float = 1.0) -> np.array:
    """
    A helper function to convert 2d state used by the pendulum gym render() function to 3d coordinate used NN learned
    system

    :param _2d_state:
    :param radius:
    :return:
    """
    x_coord, y_coord = get_x_y_from_theta(_2d_state[0], radius=radius)

    return np.array([x_coord, y_coord, _2d_state[1]])


def get_x_y_from_theta(theta_rad, radius: float = 1.0):
    x = radius * math.cos(theta_rad)
    y = radius * math.sin(theta_rad)

    return x, y


def simulate_learned_pendulum_model(model,
                                    gym_env,
                                    theta=None,
                                    init_state=None,
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
        # _3d_state = get_3d_coords_from_2_coords(curr_state)
        # next_state = model.predict(_3d_state.reshape(1, 3))
        # next_state = np.array([get_angle_from_x_y_coordinates(x_coord=next_state[0, 0], y_coord=next_state[0, 1]),
        #                        next_state[0, 2]])

        next_state = model.predict(curr_state.reshape(1, 2))

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

    # OLD PENDULUM MODELS
    # model_dirs = [
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_2500",  # model 1
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_5000", # model 2
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",  # model 3
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_7500",  # model 4
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.001_data_7500",  # model 5
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_10000",  # model 6
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.01_data_2500",
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.001_data_2500",  # model 7
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.0015_data_5000",  # model 8
    # ]

    model_dirs = [
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_2500",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_5000",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.004_data_7500"  # I like this model does well between -20 to 20
    ]

    for counter, model_path in enumerate(model_dirs):
        # load model weights
        model = tf.keras.models.load_model(filepath=model_path)

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
    max_steps = 0
    for step in range(rollout):
        # roll out
        next_state = model.predict(observation)
        # next_state = akash_model.predict(observation)
        # state_diff[step] = abs(next_state - observation)
        state_diff[step] = next_state

        if observation.shape[1] == 3:
            if next_state[0, 2] <= 0.1 or next_state[0, 2] >= 2.0:
                max_steps = step
                break

        observation = next_state

    # add function to plot in matplotlib
    # animate_nn_behavior(state_diff)

    # ylabels: list = [r"$\Delta x$", r"$\Delta y$", r"$\Delta \dot{\theta}$"]
    # ylabels: list = [r"$x$", r"$y$", r"$\dot{\theta}$"]
    if observation.shape[1] == 2:
        ylabels: list = [r"$\theta$", r"$\dot{\theta}$"]
    elif observation.shape[1] == 3:
        ylabels: list = [r"$\theta$", r"$\dot{\theta}$", r"$r$"]
    else:
        print("Invalid system dimension length for Pendulum System")
        sys.exit(-1)

    for ax_id in range(state_diff.shape[1]):
        # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        fig_axs[ax_id].plot(state_diff[:max_steps, ax_id], color, label=legend)
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
                                      system_dimension: int,
                                      color,
                                      legend,
                                      init_state=None,
                                      use_akashs_model: bool = False,
                                      model_dir_path: str = '',
                                      use_multiple_models: bool = False):
    state_space_dim = system_dimension
    if isinstance(init_state, type(None)):
        mu, sigma = 0, 0.1  # mean and standard deviation
        np.random.seed(2)
        init_state = np.random.normal(mu, sigma, size=(state_space_dim,))

    if use_multiple_models:
        # model_dirs = [
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_2500",
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_5000", # - only good in one region of thetadot
        #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_7500", # - kinda blows up at -1.0 but overall performance is good
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.0015_data_10000", - blows up
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.01_data_2500", # - kinda blows up at 1.0
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.001_data_2500", # blows up at 1.0
        #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_128_loss_0.0015_data_5000", #- only good in one region of thetadot
        # ]
        # NEW 2d pendulum models
        # model_dirs = [
        #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_2500",
        #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_5000",
        #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.01_data_7500",
        #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_neurons_64_loss_0.004_data_7500"
        #     # I like this model does well between -20 to 20
        # ]

        # NEW 3d pendulum models
        model_dirs = [
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.01_data_5000",
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.01_data_10000",
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.5_data_7500",
            # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.4_data_7500",
            "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_model_3d_neurons_64_loss_0.3_data_7500" # - best 3d pendulum model
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
                                       rollout=100)

    fig_axs[0].text(0.8, 1.2,
                    "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    transform=fig_axs[0].transAxes)

    # plt.show(block=True)


def generate_2d_pendulum_data():
    """
    A generate 2d pendulum data

    1. theta(t+1): theta(t) + thetadot(t) deltat + ((3g/2l)*(sin(theta(t)) deltat)) + ((3/ml**2)*(u*delta_t**2))
    2. thetadot(t+1): thetadot(t) + (3g/2l)*(sin(thetat)deltat) + ((3/ml**2)*(u*deltat))
    :return:
    """

    num_trajectories = 50000
    filename = f'/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/' \
               f'data/pendulum_v0_{num_trajectories}.txt'

    ''' Create a file tp dump the data in csv format '''
    with open(filename, 'w+') as file_handle:
        file_handle.write('"ip1","ip2","op1","op2"')
        file_handle.write('\n')
        file_handle.close()

    '''Generate Synthetic Data '''
    print("Performing rollouts ")
    file_handle = open(filename, 'a')

    pendulum = gym.make("Pendulum-v0")

    # define pendulum constants
    pendulum_dt = pendulum.dt
    pendulum_g = pendulum.g
    pendulum_l = pendulum.l
    pendulum_m = pendulum.m
    pendulum_max_speed = pendulum.max_speed

    pendulum_agent = PendulumAgent()

    for i in range(num_trajectories):
        # sample a theta and theta dot value
        th_ip = np.random.uniform(low=-math.pi, high=math.pi)  # theta in radians
        th_dot_ip = np.random.uniform(low=-1, high=1)

        # Convert the theta to x & y and ge the control value from it.
        _3d_state = get_3d_coords_from_2_coords(np.array([th_ip, th_dot_ip]))
        control = pendulum_agent.decide(_3d_state)[0]   # you don't have to clip it as it is always between [-2, 2]

        th_dot_op = th_dot_ip + (((3 * pendulum_g) / (2 * pendulum_l ** 2)) * (math.sin(th_ip) * pendulum_dt)) \
                    + ((3 / (pendulum_m * pendulum_l ** 2)) * (control * pendulum_dt))

        th_dot_op = np.clip(th_dot_op, -pendulum_max_speed, pendulum_max_speed)

        th_op = th_ip + (th_dot_op * pendulum_dt)

        file_handle.write(
            f'"{th_ip}","{th_dot_ip}","{th_op}","{th_dot_op}"')
        file_handle.write('\n')

    file_handle.close()


def generate_3d_pendulum_data():
    """
    A generate 2d pendulum data

    1. theta(t+1): theta(t) + thetadot(t) deltat + ((3g/2l)*(sin(theta(t)) deltat)) + ((3/ml**2)*(u*delta_t**2))
    2. thetadot(t+1): thetadot(t) + (3g/2l)*(sin(thetat)deltat) + ((3/ml**2)*(u*deltat))
    3. here l is also an input of variable length
    :return:
    """

    num_trajectories = 10000
    filename = f'/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/' \
               f'data/pendulum_v0_3d_{num_trajectories}.txt'

    ''' Create a file tp dump the data in csv format '''
    with open(filename, 'w+') as file_handle:
        file_handle.write('"ip1","ip2","ip3","op1","op2","op3"')
        file_handle.write('\n')
        file_handle.close()

    '''Generate Synthetic Data '''
    print("Performing rollouts ")
    file_handle = open(filename, 'a')

    pendulum = gym.make("Pendulum-v0")

    # define pendulum constants
    pendulum_dt = pendulum.dt
    pendulum_g = pendulum.g
    pendulum_m = pendulum.m
    pendulum_max_speed = pendulum.max_speed

    pendulum_agent = PendulumAgent()

    for i in range(num_trajectories):
        # sample a theta and theta dot value
        th_ip = np.random.uniform(low=-math.pi, high=math.pi)  # theta in radians
        th_dot_ip = np.random.uniform(low=-1, high=1)  # thetadot in radians/s
        pendulum_l = np.random.uniform(low=0.1, high=1)  # l in m

        # Convert the theta to x & y and ge the control value from it.
        _3d_state = get_3d_coords_from_2_coords(np.array([th_ip, th_dot_ip]), radius=pendulum_l)
        control = pendulum_agent.decide(_3d_state)[0]   # you don't have to clip it as it is always between [-2, 2]

        th_dot_op = th_dot_ip + (((3 * pendulum_g) / (2 * pendulum_l ** 2)) * (math.sin(th_ip) * pendulum_dt)) \
                    + ((3 / (pendulum_m * pendulum_l ** 2)) * (control * pendulum_dt))

        th_dot_op = np.clip(th_dot_op, -pendulum_max_speed, pendulum_max_speed)

        th_op = th_ip + (th_dot_op * pendulum_dt)

        file_handle.write(
            f'"{th_ip}","{th_dot_ip}","{pendulum_l}","{th_op}","{th_dot_op}","{pendulum_l}"')
        file_handle.write('\n')

    file_handle.close()


def plot_pendulum_w_cbf_evolution(system_state: np.array,
                                  mat_file_dir: str,
                                  rollout: int = 100,
                                  num_controllers: int = 2,
                                  print_flag: bool = False,
                                  use_controller: bool = False,
                                  visulaize_traj: bool = False,
                                  record_flag: bool = False,
                                  simulate: bool = False):
    """
    A helper function to plot the evolution of the states of 2d and 3d pendulum along with the additive controller
    :return:
    """

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
        # control_coeff_matrix = control_dict.get('control_matrix1')
        # updated to matrix name to  control array
        control_coeff_matrix = control_dict.get('array_control')

    partitions, partition_dim_list = get_hypercube(eps=eps.flatten(), state_space=state_space, n=s_dim, print_flag=print_flag)

    # get partitions to create the look up dictionary
    partition_from_mat = control_dict.get('partitions')

    # process it before creating the dict
    processed_partitions = postprocess_partition_dump(partition_from_mat)

    # create the hypercube lookup dict
    hypercube_idx_dict = create_hypercube_idx_dict(processed_partitions)

    # tf_pendulum_models = tf.keras.models.load_model(cartpole_dir)
    tf_pendulum_models = get_2d_pendulum_tf_model_3_cross_x_models(hidden_neurons=64,
                                                                   var_dir_path="/home/karan/Documents/research/"
                                                                                "nn_veri_w_crown/rl_train_agent/" \
                                                                                "new_pendulum_imitation_models/"
                                                                                "2d_sys/3_layer/64_neurons/" \
                                    "pendulum_model_2d_neurons_3_cross_64_loss_0.001_data_50000/variables/variables",
                                                                   input_neurons=2,
                                                                   output_neurons=2,
                                                                   print_flag=True)

    # start simulation - keep track of states
    state_evolution = np.zeros(shape=(rollout + 1, s_dim))
    control_evolution = np.zeros(shape=(rollout, num_controllers))
    curr_state = system_state.reshape(1, 2)
    state_evolution[0] = curr_state

    mu, sigma = 0, 0.1  # mean and standard deviation
    np.random.seed(2)

    # rollout trajectory
    if simulate:
        pendulum_handle = gym.make("Pendulum-v0")
        pendulum_handle.reset()
        pendulum_handle.render()

        if record_flag:
            vid = video_recorder.VideoRecorder(pendulum_handle,
                                               path='/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/'
                                                    'cbf_videos/pendulum_cbf_w_noise.mp4')

    for step in range(rollout):
        # get next state
        if use_controller:
            tmp_next_state = tf_pendulum_models.predict(curr_state) + np.random.normal(mu, sigma, size=(2,))
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
                                                        num_controllers=num_controllers,
                                                        time_step=step,
                                                        epsilon=eps[0, 0],
                                                        print_flag=False)
        else:
            next_state = tf_pendulum_models.predict(curr_state)

        state_evolution[step + 1] = next_state

        if simulate:
            pendulum_handle.unwrapped.state = next_state.reshape(2, )
            if record_flag:
                vid.capture_frame()
            pendulum_handle.render()

        curr_state = next_state.reshape(1, 2)

    if simulate:
        pendulum_handle.close()

    if visulaize_traj:
        fig, axs = plt.subplots(1)
        fig.suptitle('2d system trajectory')

        plt.plot(state_evolution[:, 0], state_evolution[:, 1], 'ro', linestyle='-.')
        plt.show(block=True)
    else:
        plot_2d_toy_sys_from_data_file(data_array=state_evolution,
                                       ylabels=[r"$\theta$ [rad]", r"$\dot{\theta}$ [rad/s]"],
                                       # ylabels=['Control Mag'],
                                       # title='Pendulum Model - Trajectory in each dimension',
                                       plot_boundary=True,
                                       # bdry_dict=dict({0: [-1, 1]})  # control min-max values for
                                       bdry_dict=dict({0: [-math.pi/15, math.pi/15], 1: [-1, 1]}),
                                       block_plot=False,
                                       save_path="/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/"
                    "pendulum_3l_w_noise/pen_state_w_noise.png"
                                       )

        # plot_from_data_file(state_evolution=state_evolution)
        plot_2d_toy_sys_from_data_file(data_array=control_evolution,
                                       # ylabels=[r"$x (m)$", r"$\dot{x} (m/s)$", r"$\theta (deg)$", r"$\dot{\theta} (deg/s)$"],
                                       ylabels=[r"$u1 $"],
                                       # title='Cartpole Model - Trajectory in each dimension',
                                       plot_boundary=False,
                                       bdry_dict=dict({0: [-1, 1]}),  # control min-max values for
                                       block_plot=False,
                                       save_path="/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/"
                    "pendulum_3l_w_noise/pen_ctrl_w_noise.png"
                                       )


if __name__ == "__main__":
    pendulum_control_mat_file = ""

    # system_state = np.array([1e-8, 1e-8])  # 2d array
    # system_state = np.array([12*(math.pi/180), 0.6])  # 2d array
    system_state = np.array([0.114, -0.3])  # 2d 3 layer pendulum worst hypercube
    # system_state = np.arrya([0, 0, 0]) # 3d array

    plot_pendulum_w_cbf_evolution(system_state=system_state,
                                  mat_file_dir="/home/karan/Documents/research/nn_veri_w_crown/"
                                               "rl_train_agent/pendulum_control_data/partition_data_120.mat",
                                  rollout=1000,
                                  num_controllers=1,
                                  print_flag=False,
                                  use_controller=True,
                                  visulaize_traj=False,
                                  record_flag=False,
                                  simulate=True)
