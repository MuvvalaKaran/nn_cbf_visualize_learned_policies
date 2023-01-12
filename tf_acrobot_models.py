import tensorflow as tf


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DIR_PATH = "/home/karan/Documents/research/nn_veri_w_crown/rl_train_agent"


def get_acrobot_tf_model_1_cross_x(hidden_neurons: int,
                                   variables_path: str,
                                   print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 2 Layers

    1.One Hidden layer with x neurons in this layer
    """

    # build the model
    input_layer = 6
    output_layer = 6

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # load model weights
    model.load_weights(filepath=variables_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model


def get_acrobot_tf_model_2_cross_x(hidden_neurons: int,
                                   variables_path: str,
                                   print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 2 Layers

    1.One Hidden layer with x neurons in this layer
    """

    # build the model
    input_layer = 6
    output_layer = 6

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_2"),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # load model weights
    model.load_weights(filepath=variables_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model


def get_acrobot_tf_model_3_cross_x(hidden_neurons: int,
                                   variables_path: str,
                                   print_flag: bool = False) -> tf.keras.models.Model:
    """
    A helper function that build the learned cartpole model in OpenAI Gym. The NN architecture is as follows:

    Architecture: Fully-connected Neural Network with 2 Layers

    1.One Hidden layer with x neurons in this layer
    """

    # build the model
    input_layer = 6
    output_layer = 6

    # NOTE: LAYERS has attribute dtype which default to float32
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_1", input_shape=(input_layer,)),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_2"),
        tf.keras.layers.Dense(hidden_neurons, activation='relu', name="dense_3"),
        tf.keras.layers.Dense(output_layer, name="predictions")
    ])

    # load model weights
    model.load_weights(filepath=variables_path)

    # when you use summary the input object is not displayed as it is not a layer.
    if print_flag:
        model.summary()

    return model