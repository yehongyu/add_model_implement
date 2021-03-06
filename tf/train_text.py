# coding=utf-8
import tensorflow as tf
import numpy as np
import explore_data
import vectorize_data
import build_model

def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """ Trains n-gram model on the given dataset.

    # Arguments
    :param data: tuples of training and test texts and labels.
    :param learning_rate: float,
    :param epochs: int, model had see total train samples as one epoch.
    :param batch_size: int, number of samples per batch, update weight after each batch.
    :param layers: int, number of Dense layers in the model.
    :param units: int, node of Dense layers
    :param dropout_ratee: percentage of input to drop at Dropout layers
    :return:
    """

    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify the validation labels are in the same range as training labels.
    num_classes = explore_data.get_num_classes(train_labels)
    #unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    #if len(unexpected_labels):
    #    raise ValueError("{unexpected_labels}".format(unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val = vectorize_data.ngram_vectorize(
        train_texts, train_labels, val_texts
    )

    # Create model instance.
    model = build_model.mlp_model(layers=layers,
                                  units=units,
                                  dropout_ratee=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with leaning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss.
    # If the loss does not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2
    )]

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=2, # Logs once per epoch.
        batch_size=batch_size
    )

    # Print results.
    history = history.history
    print('Validation accuracy:{acc}, loss:{loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]
    ))

    # Save model.
    model.save('IMDb_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'[-1]]



