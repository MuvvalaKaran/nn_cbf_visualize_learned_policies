import tensorflow as tf
import numpy as np


def _get_husky_tf_model_1_cross_x(hidden_neurons: int,
                                  activation: str,
                                  print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 2 Layers

    1.One Hidden layer with x neurons in this layer
    """

    # build the model
    input_layer = 4
    output_layer = 4

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation=activation, name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model


### for 3 layers with Relu, Relu and tanh activation
def _get_husky_tf_model_1_cross_x_model_6(hidden_neurons: int,
                                          print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 3 Layers

    1.First Hidden layer with Relu - 256
    """

    husky_dir_path = \
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/new_husky256_6/variables/variables"

    # build the model
    input_layer = 4
    output_layer = 4

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


def _get_husky_tf_model_1_cross_x_model_5(hidden_neurons: int,
                                          print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 3 Layers

    1.First Hidden layer with Relu - 256
    """

    husky_dir_path = \
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/new_husky256_5/variables/variables"

    # build the model
    input_layer = 4
    output_layer = 4

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


def _get_husky_tf_model_1_cross_x_model_8(hidden_neurons: int,
                                          print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 3 Layers

    1.First Hidden layer with Relu - 256
    """

    husky_dir_path = \
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/new_husky256_8/variables/variables"

    # build the model
    input_layer = 4
    output_layer = 4

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


def _get_husky_tf_model_1_cross_x_model_9(hidden_neurons: int,
                                          print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 3 Layers

    1.First Hidden layer with Relu - 256
    """

    husky_dir_path = \
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/new_husky256_9/variables/variables"

    # build the model
    input_layer = 4
    output_layer = 4

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


### for 3 layers with Relu, Relu and tanh activation
def _get_husky_tf_model_3_cross_x_model_3(hidden_neurons: int,
                                  print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 3 Layers

    1.First Hidden layer with Relu - 256
    1.Second Hidden layer with Relu - 256
    1.Third Hidden layer with Tanh - 256
    """

    husky_dir_path = \
        "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/new_husky256_3/variables/variables"

    # build the model
    input_layer = 4
    output_layer = 4

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_2"),
        tf.keras.layers.Dense(output_layer, activation='tanh', name="predictions")
    ])

    # load model weights
    model.load_weights(filepath=husky_dir_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model


def get_trained_husky_models(which_model: int, print_flag: bool = False):
    if which_model == 1:
        husky_dir_path =\
            "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/" \
            "new_husky256_norm1/variables/variables"

        mean_std_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/" \
                        "new_husky256_norm1-mean.npy"

        with open(mean_std_path, 'rb') as file_handle:
            weight_list = list(np.load(file_handle, allow_pickle=True))

    elif which_model == 2:
        husky_dir_path = \
            "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/husky_Models/" \
            "new_husky256_norm2/variables/variables"

        mean_std_path = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent/" \
                        "husky_Models/new_husky256_norm2-mean.npy"

        with open(mean_std_path, 'rb') as file_handle:
            weight_list = list(np.load(file_handle, allow_pickle=True))

    tf_husky_model = _get_husky_tf_model_1_cross_x(hidden_neurons=256, activation='tanh')

    # parse it into means and std
    _mean_ip = weight_list[0]
    _mean_op = weight_list[1]
    _std_ip = weight_list[2]
    _std_op = weight_list[3]

    # load model weights
    tf_husky_model.load_weights(filepath=husky_dir_path)

    if print_flag:
        tf_husky_model.summary()

    return tf_husky_model, (_mean_ip, _std_ip), (_mean_op, _std_op)