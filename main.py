import itertools
import sys
import math
import numpy as np
import pandas as pd
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

from gym.wrappers.monitoring import video_recorder
from gym.wrappers import Monitor
from copy import deepcopy
from tf_husky_models import get_trained_husky_models, _get_husky_tf_model_3_cross_x_model_3,\
    _get_husky_tf_model_1_cross_x_model_6, _get_husky_tf_model_1_cross_x_model_5,\
    _get_husky_tf_model_1_cross_x_model_8, _get_husky_tf_model_1_cross_x_model_9

from tf_husky_plotting_utls import plot_trained_husky_w_mean_and_std, plot_trained_husky_w_no_mean_and_std

from tf_pendulum_models import get_pendulum_tf_model_1_cross_x_models

from tf_pendulum_utls import simulate_learned_pendulum_model, simulate_pendulum,\
    compare_pendulum_models_stability, plot_3d_data, plot_delta_models_fixed_init_state, plot_delta_model_vs_deg_vs_thdot

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


class Agent:
    def decide(self, observation):
        position, velocity, angle, angle_velocity = observation
        action = int(3. * angle + angle_velocity > 0.)
        return action


def play_once_pendulum(init_state, env, agent, render=False, verbose=False):
    _state = []
    copied_env = deepcopy(env)
    copied_agent = deepcopy(agent)
    observation = env.reset()
    observation = init_state
    _state.append(init_state)
    episode_reward = 0.
    for step in itertools.count():
        if render:
            env.render()
        action = agent.decide(observation)
        # observation = get_one_step_op_pendulum(init_state=deepcopy(observation), env=copied_env, agent=copied_agent)
        observation, reward, done, _ = env.step(action)
        # if not np.allclose(observation, next_state, atol=0.0001):
        #     print("***************************FIX ME: Get next state() is NOT working*********************************")
        episode_reward += reward
        _state.append(observation)
        if done:
            # pass
            plt.plot(np.array(_state))
            break
    if verbose:
        print('get {} rewards in {} steps'.format(
                episode_reward, step + 1))

    return episode_reward, _state


def get_one_step_op_pendulum(init_state, env, agent):
    observation = env.reset()
    observation = init_state
    # env.unwrapped.state = init_state
    action = agent.decide(init_state)
    next_state, _, _, _ = env.step(action)
    return next_state



def get_one_step_op_cartpole(init_state, env, agent):
    env.reset()
    env.unwrapped.state = init_state
    action = agent.decide(init_state)
    next_state, _, _, _ = env.step(action)
    return next_state


def play_once_cartpole(init_state, env, agent, render=False, verbose=False):
    _state = []
    copied_env = deepcopy(env)
    env.reset()
    # observation = env.unwrapped.state
    env.unwrapped.state = init_state
    observation = init_state
    _state.append(init_state)
    episode_reward = 0.
    for step in itertools.count():
        if render:
            env.render()
        action = agent.decide(observation)
        next_state = get_one_step_op_cartpole(init_state=observation, env=copied_env, agent=agent)
        observation, reward, done, _ = env.step(action)
        if not np.allclose(observation, next_state, atol=0.0001):
            print("***************************FIX ME: Get next state() is NOT working*********************************")
        episode_reward += reward
        _state.append(observation)
        if done:
            # pass
            break
    if verbose:
        print('get {} rewards in {} steps'.format(
                episode_reward, step + 1))
    return episode_reward, _state


def generate_data(print_flag: bool = False):
    # cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500"
    # cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000"

    filename = '/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_14000.txt'

    ''' Create a file tp dump the data in csv format '''
    with open(filename, 'w') as file_handle:
        file_handle.write('"ip1","ip2","ip3","ip4","op1","op2","op3","op4"')
        file_handle.write('\n')
        file_handle.close()

    '''Generate Synthetic Data '''
    print("Performing rollouts ")
    file_handle = open(filename, 'a')

    # A, Ad = get_dynamics()
    # high_b = np.random.randint(low=1, high=11) * 1
    # low_a = np.random.randint(low=1, high=11) * -1
    atol = +0.001
    highs = [+2.4, +0.6, math.pi/15 + atol, +0.6]
    lows = [-2.4, -0.6, -math.pi/15 + atol, -0.6]
    # high_b = 10 + 0.001
    # low_a = -10

    num_trajectories = 14000
    _alpha = 0.5

    agent = Agent()
    cartpole = gym.make("CartPole-v0")

    for i in range(num_trajectories):
        ip = np.random.uniform(low=lows, high=highs, size=(4, ))

        # ge the next state
        ip_prime = get_one_step_op_cartpole(init_state=ip, env=cartpole, agent=agent)
        # old method
        # op_state = (_alpha * ((ip_prime - ip) /np.linalg.norm(ip_prime - ip))) + ip
        # new method s -> del s'
        op_state = ip_prime - ip

        file_handle.write(f'"{ip[0]}","{ip[1]}","{ip[2]}","{ip[3]}","{op_state[0]}","{op_state[1]}","{op_state[2]}","{op_state[3]}"')
        file_handle.write('\n')

    file_handle.close()


def generate_data_pendulum():
    """
    ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
    :param print_flag:
    :return:
    """

    num_trajectories = 10000
    filename = f'/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/' \
               f'data/pendulum_v0_{num_trajectories}.txt'

    ''' Create a file tp dump the data in csv format '''
    with open(filename, 'w') as file_handle:
        file_handle.write('"ip1","ip2","ip3","op1","op2","op3"')
        file_handle.write('\n')
        file_handle.close()

    '''Generate Synthetic Data '''
    print("Performing rollouts ")
    file_handle = open(filename, 'a')

    atol = +0.001
    highs = [+1, +1, 5]
    lows = [-1, -1,  -5]
    pendulum = gym.make("Pendulum-v0")
    pendulum_agent = PendulumAgent()

    for i in range(num_trajectories):
        # ip = np.random.uniform(low=lows, high=highs, size=(3, ))
        # ge the next state
        # ip_prime = get_one_step_op_pendulum(init_state=ip, env=pendulum, agent=pendulum_agent)
        ip_prime = pendulum.reset()
        action = pendulum_agent.decide(ip_prime)
        next_state, _, _, _ = pendulum.step(action)
        # old method
        # op_state = (_alpha * ((ip_prime - ip) /np.linalg.norm(ip_prime - ip))) + ip
        # new method s -> s'
        op_state = next_state

        file_handle.write(f'"{ip_prime[0]}","{ip_prime[1]}","{ip_prime[2]}","{op_state[0]}","{op_state[1]}","{op_state[2]}"')
        file_handle.write('\n')

    file_handle.close()



def _simulate_learned_cartpole_model(gym_env,
                                     init_state: np.array = np.array([-1.0, 0.25, -2*0.035, 0.2]),
                                     record_flag: bool = False):
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.01_data_2500",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_2500",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_25000",


    # 128 ones
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500"
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000"

    # cartpole_dir = '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model'

    # cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_25000"
    cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500"
    # cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000"

    ## MODEL 3
    # cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/trained_cartpole_2500_1_cross_8_loss_0001.h5"

    # cartpole_dir = '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_2500_1_cross_16_loss_001'

    # MODELS w REGULARIZATION
    model_dirs = [
        # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000",
        # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.025_data_25000"
        # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.03_data_25000"
    ]

    cartpole_model = tf.keras.models.load_model(cartpole_dir)
    gym_env.reset()
    gym_env.render()

    if record_flag:
        vid = video_recorder.VideoRecorder(gym_env, path='/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/NN_videos/video_5.mp4')

    # start simulation
    curr_state = init_state.reshape(1, 4)
    for steps in range(500):
        # get next state
        next_state = cartpole_model.predict(curr_state)
        gym_env.unwrapped.state = next_state.reshape(4, )
        if record_flag:
          vid.capture_frame()
        gym_env.render()
        curr_state = next_state

    gym_env.close()


def compare_models_stability(init_state = None):
    state_space_dim = 3
    if isinstance(init_state, type(None)):
        mu, sigma = 0, 0.1  # mean and standard deviation
        np.random.seed(2)
        init_state = np.random.normal(mu, sigma, size=(state_space_dim,))

    fig, axs = plt.subplots(state_space_dim)
    fig.suptitle('Cartpole Stabilization - only models')


    ###################### OLD MODELS with no regularization
    # Models with 8/16 neurons
    # model_dirs = ['/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/trained_cartpole.h5',
    #               '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/trained_cartpole_2500_1_cross_8_loss_0001.h5',
    #               '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_2500_1_cross_16_loss_001']

    # # Models with 32 neurons
    # model_dirs = [
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0.001_data_25000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0.001_data_250000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0.001_data_2500000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_001",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0001"]
    #
    # # Models with 64 neurons
    # model_dirs = [
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.01_data_2500",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_2500",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_25000",
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_250000",
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_2500000"
    # ]
    #
    # # Models with 128 neurons
    # model_dirs = [
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500",
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_25000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_25000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_250000",
    # # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_250000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500000",
    # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000"
    # ]

    ###################### NEW MODELS with regularization
    model_dirs=[
    "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000",
    "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.025_data_25000",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.03_data_25000"
    ]


    color_scheme: list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:brown', 'tab:cyan', 'tab:purple']
    legend_list = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']
    xytests = [(-20, 20), (-20, 30), (-20, 40), (-20, 50), (-20, 60), (-20, 70), (-20, 80)]

    for counter, _model in enumerate(model_dirs):

        _plot_state_evolution(dir_path=_model,
                              init_state=init_state.reshape(1, state_space_dim),
                              fig_axs=axs,
                              color=color_scheme[counter],
                              legend=legend_list[counter],
                              xyloc=xytests[counter])

    axs[0].text(0.8, 1.2,
                "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                transform=axs[0].transAxes)

    plt.show(block=True)


def plot_4d_data():
    # X, Y, Z = get_test_data(0.05)
    data_dir_list = [
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_5000.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_2500.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_7500.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_14000.txt",
        "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_25000.txt",
    ]

    for data_dir in data_dir_list:
        # data_dir = "/home/karan/Documents/research/nn_veri_w_crown/cbf_nn/certified_nn_bounds/data/cartpole_v0_5000.txt"
        data = pd.read_csv(data_dir)

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        X_data = data[["ip1", "ip2", "ip3", "ip4"]].values

        X = X_data[:, 0][:, np.newaxis].T
        Y = X_data[:, 1][:, np.newaxis].T
        Z = X_data[:, 2][:, np.newaxis].T
        C = X_data[:, 3][:, np.newaxis].T

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\dot{x}$")
        ax.set_zlabel(r"$\theta$")
        plt.title("Input Data visualization - Data:{}".format(X_data.shape[0]))

        img = ax.scatter(X, Y, Z, c=C, cmap=plt.get_cmap('Blues'))
        fig.colorbar(img)
    plt.show()


def compare_models_w_rl_agent(init_state = None):
    cartpole_env = gym.make("CartPole-v0")
    agent = Agent()
    np.random.seed(2)
    state_space_dim = 4

    if isinstance(init_state, type(None)):
        mu, sigma = 0, 0.1  # mean and standard deviation
        np.random.seed(2)
        init_state = np.random.normal(mu, sigma, size=(state_space_dim,))

    fig, axs = plt.subplots(state_space_dim)
    fig.suptitle('Cartpole Stabilization - models vs rl agent')

    ###################### OLD MODELS with no regularization
    # # model_dirs = ['/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/trained_cartpole.h5',
    # #               '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/trained_cartpole_2500_1_cross_8_loss_0001.h5',
    # #               '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_2500_1_cross_16_loss_001']
    #
    # model_dirs = [
    #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0.001_data_25000",
    #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0.001_data_250000",
    #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0.001_data_2500000",
    #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_001",
    #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_32_loss_0001"]
    #
    # # model_dirs = [
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.01_data_2500",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_2500",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_25000",
    # #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_250000"
    # #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.001_data_2500000"
    # # ]
    #
    # # Models with 128 neurons
    # # model_dirs = [
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500",
    # #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_25000",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_25000",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_250000",
    # #     # "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_250000",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500000",
    # #     "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000"
    # # ]

    ###################### NEW MODELS with regularization
    model_dirs = [
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.025_data_25000",
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_64_loss_0.03_data_25000"
    ]

    color_scheme: list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red',  'tab:brown', 'tab:cyan', 'tab:purple']
    legend_list = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6', 'model7']
    xytests = [(-20, 20), (-20, 30), (-20, 40), (-20, 50), (-20, 60), (-20, 70), (-20, 80)]

    for counter, _model in enumerate(model_dirs):
        # this code for plotting the difference in the RL agent and NN model at each time step
        # _plot_state_evolution_difference_w_rlagent(agent=agent,
        #                                            rl_cartpole_agent=cartpole_env,
        #                                            dir_path=_model,
        #                                            init_state=init_state.reshape(1, 4),
        #                                            fig_axs=axs,
        #                                            color=color_scheme[counter],
        #                                            legend=legend_list[counter],
        #                                            xyloc=xytests[counter])

        # this code for plotting trajectories
        _plot_state_evolution_w_rlagent(dir_path=_model,
                                        init_state=init_state.reshape(1, 4),
                                        fig_axs=axs,
                                        color=color_scheme[counter],
                                        legend=legend_list[counter],
                                        xyloc=xytests[counter])


    # add rl trajectory to the plot
    observation = init_state
    rl_op = np.zeros(shape=(500, 4))
    for step in range(500):
        # get next state from RL agent
        op = get_one_step_op_cartpole(init_state=observation.reshape(4, ), env=cartpole_env, agent=agent)
        rl_op[step] = op.reshape(1, 4)
        observation = op

    for ax_id in range(state_space_dim):
        axs[ax_id].plot(rl_op[:, ax_id], '--k', label='rl_agent')

    axs[0].text(0.8, 1.2,
                "Init State:" + ', '.join(["'{:.{}f}'".format(_s, 5) for _s in init_state]),
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                transform=axs[0].transAxes)

    plt.show(block=True)


def _plot_state_evolution_difference_w_rlagent(agent,
                                               rl_cartpole_agent,
                                               dir_path,
                                               init_state,
                                               fig_axs,
                                               color,
                                               legend,
                                               xyloc: tuple = (-20, 20)):
    """
    A helper function to plot the evolution to changge in each state in the cartpole model
    :return:
    """

    cartpole_model = tf.keras.models.load_model(dir_path)
    observation = init_state
    # col = stat diff # row = time step
    state_diff = np.zeros(shape=(500, 4))
    rl_op = np.zeros(shape=(500, 4))
    for step in range(500):
        # get next state from NN
        next_state = cartpole_model.predict(observation)

        # get next state from RL agent
        op = get_one_step_op_cartpole(init_state=observation.reshape(4, ), env=rl_cartpole_agent, agent=agent)
        rl_op[step] = op.reshape(1, 4)
        # state_diff[step] = abs(next_state - op.reshape(1, 4))
        state_diff[step] = next_state - op.reshape(1, 4)
        observation = next_state

    ylabels: list = [r"$\Delta x$", r"$\Delta \dot{x}$", r"$\Delta \theta$", r"$\Delta \dot{\theta}$"]

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


def _plot_state_evolution_w_rlagent(dir_path,
                                    init_state,
                                    fig_axs,
                                    color,
                                    legend,
                                    xyloc: tuple = (-20, 20)):
    """
    A helper function to plot the evolution to changge in each state in the cartpole model
    :return:
    """

    cartpole_model = tf.keras.models.load_model(dir_path)
    observation = init_state
    # col = stat diff # row = time step
    states = np.zeros(shape=(500, 4))
    rl_op = np.zeros(shape=(500, 4))
    for step in range(500):
        # get next state from NN
        next_state = cartpole_model.predict(observation)
        states[step] = next_state
        observation = next_state

    ylabels: list = [r"$\Delta x$", r"$\Delta \dot{x}$", r"$\Delta \theta$", r"$\Delta \dot{\theta}$"]

    for ax_id in range(states.shape[1]):
        # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
        fig_axs[ax_id].plot(states[:, ax_id], color, label=legend)
        fig_axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
        fig_axs[ax_id].legend(loc='best')

        # The key option here is `bbox`. I'm just going a bit crazy with it.
        fig_axs[ax_id].annotate('{:.{}f}'.format(states[0, ax_id], 5), xy=(1, states[0, ax_id]),
                                xytext=xyloc,
                                textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                color='red'))

    plt.plot()


def _plot_state_evolution(dir_path, init_state, fig_axs, color, legend, xyloc: tuple = (-20, 20), rollout: int = 100):
    """
    A helper function to plot the evolution to chnage in each state in the cartpole model
    :return:
    """
    cartpole_model = tf.keras.models.load_model(dir_path)

    observation = init_state
    # col = stat diff # row = time step
    state_diff = np.zeros(shape=(rollout, observation.shape[1]))
    for step in range(rollout):
        # roll out
        next_state = cartpole_model.predict(observation)
        state_diff[step] = abs(next_state - observation)
        observation = next_state

    ylabels: list = [r"$\Delta x$", r"$\Delta \dot{x}$", r"$\Delta \theta$", r"$\Delta \dot{\theta}$"]

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


def plot_pendulum_models():
    tf_pendulum_62_hidden_model_1 = get_pendulum_tf_model_1_cross_x_models(hidden_neurons=64,
                                                                           which_model=1,
                                                                           print_flag=True)

    roll_out_steps = 20
    init_state = np.array([0.0, 0.0, 0.1])
    init_list = [init_state]
    state_space_dim = 3

    fig, axs = plt.subplots(state_space_dim)
    fig.suptitle('Cartpole Stabilization - models vs rl agent')

    for init_state in init_list:
        _states = np.zeros(shape=(roll_out_steps, state_space_dim))
        _states[0] = init_state

        for step in range(roll_out_steps-1):
            # get next state from NN
            next_state = tf_pendulum_62_hidden_model_1.predict(_states[step].reshape(1, 3))
            _states[step+1] = next_state

        ylabels: list = [r"$\Delta x$", r"$\Delta \dot{x}$", r"$\Delta \theta$", r"$\Delta \dot{\theta}$"]

        for ax_id in range(_states.shape[1]):
            # fig_axs[ax_id].plot(state_diff[:, ax_id], color_scheme[ax_id])
            axs[ax_id].plot(_states[:, ax_id], 'tab:green', label='Model 64- 1')
            axs[ax_id].set(xlabel='time-step', ylabel=ylabels[ax_id])
            axs[ax_id].legend(loc='best')

            # The key option here is `bbox`. I'm just going a bit crazy with it.
            axs[ax_id].annotate('{:.{}f}'.format(_states[0, ax_id], 5), xy=(1, _states[0, ax_id]),
                                                 xytext=(-20, 20),
                                                 textcoords='offset points', ha='center', va='bottom',
                                                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                                color='red'))

        plt.plot()

    plt.show(block=True)



def plot_learned_sys_phase_portrait(scale=1.0):
    """"
    A function to plot the origina phase portrait of the system

    """
    # Model 3
    cartpole_dir = '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_2500_1_cross_16_loss_001'

    # Model5
    cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500"


    cartpole_model = tf.keras.models.load_model(cartpole_dir)
    use_leraned_nn: bool = True
    lin_x = np.arange(-12*math.pi/180, 12*math.pi/180, 0.5*math.pi/180)
    lin_y = np.arange(-0.5, 0.5, 0.1)

    cartpole = gym.make("CartPole-v0")
    agent = Agent()

    # _fig, _ax = plt.subplots(subplot_kw=dict(aspect='equal'))
    _fig, _ax = plt.subplots()

    for i in lin_x:
        for j in lin_y:
            curr_state = np.array([[0.0, 0.0, i, j]])
            if use_leraned_nn:
                op = cartpole_model.predict(curr_state)

                _ax.arrow(x=curr_state[0][2] * 180/math.pi,
                          y=curr_state[0][3],
                          dx=scale * (op[0][2] - curr_state[0][2]) * 180/math.pi,
                          dy=scale * (op[0][3] - curr_state[0][3]),
                          color='red',
                          head_width=0.02,
                          length_includes_head=True,
                          alpha=0.5)
            else:
                op = get_one_step_op_cartpole(init_state=curr_state.reshape(4, ), env=cartpole, agent=agent)

                _ax.arrow(x=curr_state[0][2]* 180/math.pi,
                          y=curr_state[0][3],
                          dx=180/math.pi * (op[2] - curr_state[0][2]),
                          dy=(op[3] - curr_state[0][3]),
                          color='red',
                          # head_width=0.1,
                          # length_includes_head=True,
                          alpha=0.5)

    plt.plot()

    # Model 2
    cartpole_dir = '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/trained_cartpole_2500_1_cross_8_loss_0001.h5'
    # Model4
    cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000"
    cartpole_model = tf.keras.models.load_model(cartpole_dir)

    for i in lin_x:
        for j in lin_y:
            curr_state = np.array([[0.0, 0.0, i, j]])
            if use_leraned_nn:
                op = cartpole_model.predict(curr_state)

                _ax.arrow(x=curr_state[0][2] * 180/math.pi,
                          y=curr_state[0][3],
                          dx=scale * (op[0][2] - curr_state[0][2]) * 180/math.pi,
                          dy=scale * (op[0][3] - curr_state[0][3]),
                          color='green',
                          head_width=0.02,
                          length_includes_head=True,
                          alpha=0.5)
            else:
                op = get_one_step_op_cartpole(init_state=curr_state.reshape(4, ), env=cartpole, agent=agent)

                _ax.arrow(x=curr_state[0][2]* 180/math.pi,
                          y=curr_state[0][3],
                          dx=180/math.pi * (op[2] - curr_state[0][2]),
                          dy=(op[3] - curr_state[0][3]),
                          color='green',
                          # head_width=0.1,
                          # length_includes_head=True,
                          alpha=0.5)


    # _ax.set_aspect('equal', 'box')
    _ax.set_xlabel('theta')
    _ax.set_ylabel('dtheta')
    plt.show(block=True)

    return _fig, _ax


def _dump_model_weights():
    # Husky Model 5
    # husky_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/new_husky256_5/variables/variables"
    # imitation_husky_model = tf.keras.models.load_model(husky_dir)
    tf_model_model_5 = _get_husky_tf_model_1_cross_x_model_5(hidden_neurons=256)

    tf_model_model_5.summary()
    weight_list = tf_model_model_5.get_weights()
    tf_model_model_5.save("husky_model_5.h5")

    cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.01_data_2500"  # model5
    # cartpole_dir = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_1_cross_128_loss_0.001_data_2500000" # model4


    # cartpole_dir = '/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/cartpole_model_2500_1_cross_16_loss_001'
    # cartpole_model = tf.keras.models.load_model(cartpole_dir)

    # cartpole_model.save_weights("trained_ckpt")
    # cartpole_model.summary()
    # weight_list = cartpole_model.get_weights()
    # dump weights using numpy
    with open('husky_model_5.npy', 'wb') as f:
        np.save(f, weight_list)

    # cartpole_model.save("husky_model_5.h5")


if __name__ == "__main__":
    record_flag: bool = False
    simuale_rl: bool = False
    simulate_nn: bool = False
    save_nn: bool = False
    generate_data_flag: bool = False
    generate_pendulum_data_flag: bool = False
    plot_pendulum_data_flag: bool = False
    plot_husky: bool = False
    plot_pendulum: bool = True
    simulate_pendulum_flag: bool = False

    # _plot_state_evolution()
    # init_state = np.array([0, 0, 0.035, 0])
    # pendulum = gym.make("Pendulum-v0")
    # init_state = pendulum.reset()
    # compare_models_stability(init_state=init_state)
    # compare_models_w_rl_agent(init_state=init_state)
    # plot_4d_data()
    # sys.exit(-1)

    # plot_learned_sys_phase_portrait()
    if plot_husky:
        # plot_trained_husky_w_mean_and_std()
        plot_trained_husky_w_no_mean_and_std()
        sys.exit(-1)

    elif plot_pendulum:
        plot_delta_models_fixed_init_state()
        # plot_delta_model_vs_deg_vs_thdot(save_flag=False)

        sys.exit(-1)

        # pendulum = gym.make("Pendulum-v0")
        # pendulum_agent = PendulumAgent()
        #
        # fig, axs = plt.subplots(3)
        # fig.suptitle('Pendulum - v0')
        # fig.set_size_inches(18.5, 10.5, forward=True)
        # for run in range(5):
        #
        #     _, _state = play_once_pendulum(init_state=np.array([0, 0, 0.0]),
        #                                    env=pendulum,
        #                                    agent=pendulum_agent,
        #                                    render=True,
        #                                    verbose=True)
        #
        #     for dim in range(3):
        #         axs[dim].plot(np.array(_state)[:, dim], label=run)
        #         axs[dim].legend(loc='best')
        #
        # plt.show(block=True)

        # pendulum_model = get_pendulum_tf_model_1_cross_x_models(hidden_neurons=128,
        #                                                         which_model=2)
        #
        # simulate_learned_pendulum_model(model=pendulum_model,
        #                                 gym_env=pendulum,
        #                                 init_state=np.array([0.99, 0.0, 0.0]).reshape(3,),
        #                                 record_flag=True,
        #                                 rollout_steps=500)
        sys.exit(-1)
        # plot_pendulum_models()

    elif plot_pendulum_data_flag:
        plot_3d_data()

    elif simulate_pendulum_flag:
        pendulum = gym.make("Pendulum-v0")
        simulate_pendulum(pendulum)

    elif save_nn:
        _dump_model_weights()
        sys.exit(-1)

    elif simuale_rl:
        if record_flag:
            cartpole = Monitor(gym.make("CartPole-v0"), './video', force=True)
        else:
            cartpole = gym.make("CartPole-v0")
        agent = Agent()
        fig, axs = plt.subplots(3)
        fig.suptitle('Pendulum - v0')
        fig.set_size_inches(18.5, 10.5, forward=True)
        for ep in range(5):
            mu, sigma = 0, 0.1  # mean and standard deviation
            init_state = np.random.normal(mu, sigma, size=(4, ))
            _, _state = play_once_cartpole(init_state=np.array([0, 0, 0.1, 0]),
                                           env=cartpole,
                                           agent=agent,
                                           render=True,
                                           verbose=True)
            cartpole.close()

            for dim in range(3):
                axs[dim].plot(np.array(_state)[:, dim], label=dim)
                axs[dim].legend(loc='best')

        plt.show()

    elif simulate_nn:
        cartpole = gym.make("CartPole-v0")
        _simulate_learned_cartpole_model(gym_env=cartpole,
                                         record_flag=record_flag)
    elif generate_data_flag:
        generate_data()

    elif generate_pendulum_data_flag:
        generate_data_pendulum()


