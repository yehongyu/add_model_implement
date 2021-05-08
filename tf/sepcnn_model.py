# coding=utf-8

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

from utils import get_last_layer_units_and_activation

def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):
    """ Creates an instance of a separable CNN model.

    :param blocks: int, number of pairs of sepCNN and pooling blocks in the model.
    :param filters: int, output dimension of the layers.
    :param kernel_size: int, length of the convolution window.
    :param embedding_dim: int, dimension of the embedding vectors.
    :param dropout_rate:
    :param pool_size:
    :param input_shape:
    :param num_classes:
    :param num_features:
    :param use_pretrained_embedding:
    :param is_embedding_trainable:
    :param embedding_matrix:
    :return: A sepCNN model instance.
    """
    op_uniits, op_activation = get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()

    # Add embedding layer. If pre-trained embedding is used add weights to
    # the embedding layer and set trainable to input is_embedding_trainable flag
    if use_pretrained_embedding:
        model.add(Embedding(
            input_dim=num_features,
            output_dim=embedding_dim,
            input_length=input_shape[0],
            weights=[embedding_matrix],
            trainable=is_embedding_trainable
        ))
    else:
        model.add(Embedding(
            input_dim=num_features,
            output_dim=embedding_dim,
            input_length=input_shape[0]
        ))
    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model


