import tensorflow as tf
import numpy as np


def get_leader_board_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(
            "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_models/Pendulum-v0.ckpt-660000.meta")
        new_saver.restore(sess,
                          "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_models/"
                          "Pendulum-v0.ckpt-660000")

        vars = tf.trainable_variables()


def get_pendulum_tf_model_1_cross_x_models(hidden_neurons: int,
                                           which_model: int = 1,
                                           print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned pendulum model in OpenAI Gym. The NN architecture is as follows:
    """

    # get_leader_board_model()

    husky_dir_path = \
            f"/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/pendulum_models/pendulum_{hidden_neurons}" \
            f"_{which_model}/variables/variables"

    # build the model
    input_layer = 3
    output_layer = 3

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # load model weights
    model.load_weights(filepath=husky_dir_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model


def get_2d_pendulum_tf_model_3_cross_x_models(hidden_neurons: int,
                                              var_dir_path: str,
                                              input_neurons: int = 2,
                                              output_neurons: int = 2,
                                              print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned pendulum model in OpenAI Gym. The NN architecture is as follows:
    """

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_neurons,)),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_2"),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_3"),
        tf.keras.layers.Dense(output_neurons, name="predictions")
    ])

    # load model weights
    model.load_weights(filepath=var_dir_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model