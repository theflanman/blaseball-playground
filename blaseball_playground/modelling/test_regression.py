import datetime

import numpy as np
import sklearn.naive_bayes
import sklearn.model_selection

import matplotlib.pyplot as plt

import wandb

from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kerasLayers
from tensorflow.keras import regularizers as kerasRegs

import blaseball_playground.data.get_games

#
wandb.init(project="test_regression")


class Expando(object):
    pass


# wandb = Expando()
# wandb.config = Expando()

wandb.config.batch_size = int(2**14)
wandb.config.epochs = 10000
wandb.config.learning_rate = 1e-6
wandb.config.layer1activation = "relu"
wandb.config.layer2activation = "relu"
wandb.config.layer3activation = None

game_data = blaseball_playground.data.get_games.main()

y = np.array(game_data[['away_score', 'home_score']])
# y = np.diff(y)
x = np.array(game_data[[c for c in game_data.columns if c not in ['away_score', 'home_score']]])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x.astype(np.float32), y.astype(np.float32))

log_dir = ".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = keras.Sequential(
    [
        kerasLayers.Input(shape=(x.shape[1]), name="input"),
        kerasLayers.Dense(80, activation=wandb.config.layer1activation, name="layer1"),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        kerasLayers.Dropout(0.5),
        kerasLayers.Dense(80, activation=wandb.config.layer2activation, name="layer2"),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(25, activation="sigmoid", name="layer3", kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(25, activation="sigmoid", name="layer4", kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        kerasLayers.Dense(y.shape[1], activation=wandb.config.layer3activation, name="output"),   #, kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        kerasLayers.Lambda(lambda x: x*100, output_shape=(y.shape[1],), name="scaling")
    ]
)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=wandb.config.learning_rate),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.MeanSquaredError(),
                       keras.metrics.RootMeanSquaredError()])

history = model.fit(x_train,
                    y_train,
                    batch_size=wandb.config.batch_size,
                    epochs=wandb.config.epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback, WandbCallback()],
                    shuffle=True,
                    )

yp = model.predict(x)
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

error = y - yp
error_train = y_train - yp_train
error_test = y_test - yp_test

plt.scatter(error_test[:, 0], error_test[:, 1])

a = 1
