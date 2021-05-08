# coding=utf-8

def get_last_layer_units_and_activation(num_classes):
    """ Gets units and activation func for the last netword layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, acitvation func
    """

    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

