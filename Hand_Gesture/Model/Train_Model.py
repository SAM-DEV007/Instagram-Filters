from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

import os


if __name__ == '__main__':
    # Save paths
    dataset = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Dataset\\', 'Training_Data.csv')
    model_save = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model_Data\\', 'Model.hdf5')
    tflite_save = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model_Data\\', 'Model.tflite')

    # Dataset
    x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

    # Train and Test data
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state = 42)

    # Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.summary()

    # Callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save, verbose=1, save_weights_only=False)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback, es_callback]
    )

    # Evaluate model
    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)

    # Saves the model
    model.save(model_save, include_optimizer=False)

    # Converts to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save, 'wb').write(tflite_quantized_model)
