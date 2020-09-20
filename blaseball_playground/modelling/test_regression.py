import datetime

import numpy as np
import sklearn.naive_bayes
import sklearn.model_selection

import wandb
wandb.init(project="test_regression")
wandb.config.batch_size = int(2**14)
wandb.config.epochs = 10000
wandb.config.learning_rate = 1e-3

import matplotlib as mpl
mpl.use('tkagg')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kerasLayers

import blaseball_playground.data.get_games

game_data = blaseball_playground.data.get_games.main()

y = np.array(game_data[['away_score', 'home_score']])
y = np.diff(y)
x = np.array(game_data[[c for c in game_data.columns if c not in ['away_score', 'home_score']]])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x.astype(np.float32), y.astype(np.float32)/100)

log_dir = ".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = keras.Sequential(
    [
        kerasLayers.Input(shape=(x.shape[1])),
        kerasLayers.Dense(150, activation="tanh", name="layer1"),   #, kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        kerasLayers.Dense(75, activation="tanh", name="layer2"),   #, kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(25, activation="sigmoid", name="layer3", kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dense(25, activation="sigmoid", name="layer4", kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        kerasLayers.Dense(y.shape[1], activation=None, name="layer5"),   #, kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
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
                    callbacks=[tensorboard_callback],
                    shuffle=True,
                    )

a = 1
