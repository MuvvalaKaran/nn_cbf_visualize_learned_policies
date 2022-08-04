import gym
import time
import math
import numpy as np
import tensorflow as tf

from scipy.io import loadmat
from gym.wrappers.monitoring import video_recorder
from tf_toy_system_utls import plot_2d_toy_sys_from_data_file
from gen_utls import simulate_zamani_ex_1, create_hypercube_idx_dict, evolve_according_to_controller,\
    postprocess_partition_dump, get_hypercube


DIR_PATH = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent"


def _get_husky_tf_model_1_cross_x_no_processing(hidden_neurons: int,
                                                husky_dir_path: str,
                                                input_layer: int = 4,
                                                output_layer: int = 4,
                                                print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 2 Layers

    1.First Hidden layer with Relu - 256
    """

    # NOTE: LAYERS has attribute dtype which default to float32
    tf_husky_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # load model weights
    tf_husky_model.load_weights(filepath=husky_dir_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        tf_husky_model.summary()

    return tf_husky_model


def _get_husky_tf_model_2_cross_x_no_processing(hidden_neurons: int,
                                                husky_dir_path: str,
                                                input_layer: int = 4,
                                                output_layer: int = 4,
                                                print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 2 Layers

    1.First Hidden layer with Relu - 256
    """
    
    # used to compute bounds
    # husky_dir_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
    #                  "husky_2_layer_models/husky_256_256_1/variables/variables"

    # husky_dir_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
    #                  "husky_2_layer_models/husky_256_256_2/variables/variables"

    # NOTE: LAYERS has attribute dtype which default to float32
    tf_husky_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_2"),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # load model weights
    tf_husky_model.load_weights(filepath=husky_dir_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        tf_husky_model.summary()

    return tf_husky_model


def simulate_cartpole_w_cbf(system_state: np.array,
                            mat_file_dir: str,
                            rollout: int = 100,
                            num_controllers: int = 2,
                            print_flag: bool = False,
                            use_controller: bool = False,
                            visualize: bool = False,
                            record_flag: bool = False,
                            simulate: bool = False,
                            use_husky_5d: bool = True):
    """
    A helper function to simulate the husky behavior w cbf
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
        control_coeff_matrix = control_dict.get('array_control')

    _, partition_dim_list = get_hypercube(eps=eps[0, :], state_space=state_space, n=s_dim, print_flag=print_flag)

    # get partitions to create the look up dictionary
    partition_from_mat = control_dict.get('partitions')

    # process it before creating the dict
    processed_partitions = postprocess_partition_dump(partition_from_mat)

    # create the hypercube lookup dict
    hypercube_idx_dict = create_hypercube_idx_dict(processed_partitions)

    husky_model = _get_husky_tf_model_1_cross_x_no_processing(hidden_neurons=256,
                                                                 husky_dir_path="/home/karan/Documents/research/"
                                                                                "nn_veri_w_crown/rl_train_agent/"
                                                                                "husky_Models/new_husky256_5/"
                                                                                "variables/variables",
                                                                 print_flag=True)

    # use 2 layer model 4d
    # husky_model = _get_husky_tf_model_2_cross_x_no_processing(hidden_neurons=256,
    #                                                           husky_dir_path="/home/karan/Documents/research/"
    #                                                                          "nn_veri_w_crown/rl_train_agent/"
    #                                                                          "husky_2_layer_models/"
    #                                                                          "husky_256_256_1/variables/variables",
    #                                                           print_flag=True)

    # All 5d Husky models - Trained with 5000 data
    if use_husky_5d:
        assert use_controller is False, "Can not use controller for 5d Husky Yet"
        model_dirs = [
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_1layer_256_5000/variables/variables",
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_1layer_256_10000/variables/variables",
            # nice smooth behavior
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_1layer_512_5000/variables/variables",
            # nice smooth behavior but goes out of range quickly
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_1layer_512_10000/variables/variables",
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_2layer_256_256_5000/variables/variables",
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_2layer_256_256_10000/variables/variables",
            # nice smooth behavior but goes out of bound quickly
            # "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_2layer_512_512_5000/variables/variables",
            # nice smooth behavior but goes out of bound quickly but worse beahvior than 5000 one
            "/home/karan/Documents/research/husky-5d/husky-5d-imiation/husky-5d/husky_2layer_512_512_10000/variables/variables"
        ]

        husky_model = _get_husky_tf_model_1_cross_x_no_processing(hidden_neurons=512,
                                                                  input_layer=5,
                                                                  output_layer=5,
                                                                  print_flag=True,
                                                                  husky_dir_path=model_dirs[0])

        # husky_model = _get_husky_tf_model_2_cross_x_no_processing(hidden_neurons=512,
        #                                                           input_layer=5,
        #                                                           output_layer=5,
        #                                                           print_flag=True,
        #                                                           husky_dir_path=model_dirs[0])

    # rollout trajectory

    # start simulation - keep track of states
    # tmp overide s_dim to 5
    s_dim = 4
    state_evolution = np.zeros(shape=(rollout+1, s_dim))
    control_evolution = np.zeros(shape=(rollout, 2))
    curr_state = system_state.reshape(1, s_dim)
    state_evolution[0] = curr_state

    for step in range(rollout):
        # get next state
        if use_controller:
            # tmp_next_state = cartpole_model.predict(curr_state)

            # for 1 layers that are trained over drift
            delta = husky_model.predict(curr_state)
            tmp_next_state = delta + curr_state

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
                                                        print_flag=False,
                                                        use_husky=True)

        else:
            # next_state = cartpole_model.predict(curr_state)
            # for 1 layer husky that are trained over drift
            delta = husky_model.predict(curr_state)
            next_state = delta + curr_state

            # for 2 layer husky model - they predict next state
            # next_state = husky_model.predict(curr_state)

        state_evolution[step+1] = next_state

        curr_state = next_state.reshape(1, s_dim)

    if visualize:
        # 4d husky
        if not use_husky_5d:
            # plot_from_data_file(state_evolution=state_evolution)
            plot_2d_toy_sys_from_data_file(data_array=control_evolution,
                                           # ylabels=[r"$x (m)$", r"$\dot{x} (m/s)$", r"$\theta (deg)$", r"$\dot{\theta} (deg/s)$"],
                                           ylabels=[r"$u1 $", "$u2 $"],
                                           # title='Husky Model - Trajectory in each dimension',
                                           plot_boundary=False,
                                           bdry_dict=dict({0: [-1, 1], 1: [-1, 1]})  # control min-max values for
                                           # bdry_dict=dict({0: [-1, 1], 2: [-math.pi / 12, math.pi / 12]})
                                           )

            plot_2d_toy_sys_from_data_file(data_array=state_evolution,
                                           ylabels=[r"$x (m)$", r"$y (m)$", r"$\theta$", r"$v (m/s)$"],
                                           # ylabels=[r"$u1 $", "$u2 $"],
                                           # title='Husky Model - Trajectory in each dimension',
                                           plot_boundary=True,
                                           # bdry_dict=dict({0: [-10, 10], 1: [-10, 10]})  # control min-max values for
                                           bdry_dict=dict({0: [-0.5, 1],
                                                           2: [-math.pi / 12, math.pi / 12],
                                                           1: [-1, 1],
                                                           3: [-0.5, 0.5]})
                                           )
        # 5d husky
        else:
            # plot_2d_toy_sys_from_data_file(data_array=control_evolution,
            #                                # ylabels=[r"$x (m)$", r"$\dot{x} (m/s)$", r"$\theta (deg)$", r"$\dot{\theta} (deg/s)$"],
            #                                ylabels=[r"$u1 $", "$u2 $"],
            #                                title='Husky Model - Trajectory in each dimension',
            #                                plot_boundary=False,
            #                                bdry_dict=dict({0: [-1, 1], 1: [-1, 1]})  # control min-max values for
            #                                # bdry_dict=dict({0: [-1, 1], 2: [-math.pi / 12, math.pi / 12]})
            #                                )

            plot_2d_toy_sys_from_data_file(data_array=state_evolution,
                                           ylabels=[r"$x (m)$", r"$y (m)$", r"$\theta$", r"$v (m/s)$", r"$\omega$"],
                                           # ylabels=[r"$u1 $", "$u2 $"],
                                           title='Husky Model - Trajectory in each dimension',
                                           plot_boundary=False,
                                           # bdry_dict=dict({0: [-10, 10], 1: [-10, 10]})  # control min-max values for
                                           bdry_dict=dict({0: [-0.5, 1],
                                                           2: [-math.pi / 12, math.pi / 12],
                                                           1: [-1, 1],
                                                           3: [-0.5, 0.5]})
                                           )

    # ''' Save the model in the data folder '''
    # _dump_model = input("Do you want to save the strategy,Enter: Y/y")
    #
    # # save learned model
    # if _dump_model == "y" or _dump_model == "Y":
    #     # 0 - year; 1 - month ; 2 -day; 3-hour, 4-min, 5-sec
    #     time_struct = time.localtime(time.time())
    #     file_name = f"pendulum_control_data/cartpole_model_5_rollout_{rollout}_{time_struct[1]}_{time_struct[2]}" \
    #                            f"_{time_struct[3]}_{time_struct[4]}_{time_struct[5]}.npy"
    #
    #     file_path = DIR_PATH + "/" + file_name
    #     # dump states using numpy
    #     with open(file_name, 'wb') as file_handle:
    #         np.save(file_handle, state_evolution)


if __name__ == "__main__":

    husky_control_mat_file = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
                             "husky_control_data/imitation_husky_partition_data_900.mat"


    # 4d Husky
    # state_list = [
    #                 # np.array([0, 0, -5*math.pi/180, 0]),
    #               # np.array([0, 0, -3*math.pi/180, 0]),
    #               np.array([0.0, 0.0, 0.0, 0.0]),
    #               # np.array([0, 0, 1*math.pi/180, 0]),
    #               # np.array([0, 0, 3*math.pi/180, 0]),
    #               # np.array([0, 0, 5*math.pi/180, 0])
    #               ]
    # _system_state = np.array([0, 0, -5*math.pi/180, 0])
    # 5d husky
    state_list = [
        np.array([0.1, 0.0, 0.0, 0.0]),
        # np.array([0.0, 0.1, 0.0, 0.0, 0.0]),
        # np.array([0.1, 0.1, 0.0, 0.0, 0.0]),
        # np.array([-0.1, 0.0, 0.0, 0.0, 0.0]),
        # np.array([0.0, -0.1, 0.0, 0.0, 0.0]),
        # np.array([-0.1, -0.1, 0.0, 0.0, 0.0])
    ]

    for _system_state in state_list:
        # simulate cartpole behavior
        simulate_cartpole_w_cbf(system_state=_system_state,
                                mat_file_dir=husky_control_mat_file,
                                print_flag=True,
                                use_controller=False,
                                visualize=True,
                                rollout=150,
                                num_controllers=1,
                                use_husky_5d=False)