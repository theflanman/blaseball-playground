import blaseball_playground.config_matplotlib

import datetime

import numpy as np
import sklearn.naive_bayes
import sklearn.model_selection

import matplotlib.pyplot as plt

import wandb

import pandas as pd

from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kerasLayers
from tensorflow.keras import regularizers as kerasRegs

import blaseball_playground.data.get_games
from blaseball_playground.modelling.common import get_data

#
project_name = "test_regression"
wandb.init(project=project_name)


class Expando(object):
    pass


# wandb = Expando()
# wandb.config = Expando()

wandb.config.batch_size = 64
wandb.config.epochs = 100
wandb.config.learning_rate = 1e-4
wandb.config.layer1activation = 'sigmoid'
wandb.config.layer2activation = 'sigmoid'
wandb.config.layer3activation = 'sigmoid'

# game_data = blaseball_playground.data.get_games.main()
# game_data = pd.read_csv('./blaseball_playground/data/files/forbidden_game_stats.csv')
# game_data = game_data[[c for c in game_data.columns if 'Unnamed' not in c]]
# game_data = game_data[[c for c in game_data.columns if 'offense' not in c]]
#
# y = np.array(game_data[['away_score', 'home_score']])
# y = np.diff(y)
# x = np.array(game_data[[c for c in game_data.columns if c not in ['away_score', 'home_score']]])

dep_vars = ['called_strike', 'swinging_strike', 'ball', 'foul', 'hit']
ind_vars = ['batter_anticapitalism', 'batter_base_thirst', 'batter_buoyancy',
            'batter_chasiness', 'batter_coldness', 'batter_continuation',
            'batter_divinity', 'batter_ground_friction', 'batter_indulgence',
            'batter_laserlikeness', 'batter_martyrdom', 'batter_moxie',
            'batter_musclitude', 'batter_omniscience', 'batter_overpowerment',
            'batter_patheticism', 'batter_ruthlessness', 'batter_shakespearianism',
            'batter_suppression', 'batter_tenaciousness', 'batter_thwackability',
            'batter_tragicness', 'batter_unthwackability', 'batter_watchfulness',
            'batter_pressurization', 'batter_cinnamon', 'batter_batting_rating',
            'batter_baserunning_rating', 'batter_defense_rating',
            'batter_pitching_rating', 'pitcheranticapitalism', 'pitcherbase_thirst',
            'pitcherbuoyancy', 'pitcherchasiness', 'pitchercoldness',
            'pitchercontinuation', 'pitcherdivinity', 'pitcherground_friction',
            'pitcherindulgence', 'pitcherlaserlikeness', 'pitchermartyrdom',
            'pitchermoxie', 'pitchermusclitude', 'pitcheromniscience',
            'pitcheroverpowerment', 'pitcherpatheticism', 'pitcherruthlessness',
            'pitchershakespearianism', 'pitchersuppression', 'pitchertenaciousness',
            'pitcherthwackability', 'pitchertragicness', 'pitcherunthwackability',
            'pitcherwatchfulness', 'pitcherpressurization', 'pitchercinnamon',
            'pitcherbatting_rating', 'pitcherbaserunning_rating',
            'pitcherdefense_rating', 'pitcherpitching_rating']

merged_events = pd.read_hdf('./blaseball_playground/data/files/events/events.hdf5', 'merged_events')

x = merged_events[ind_vars]
y = merged_events[dep_vars]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)

log_dir = ".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = keras.Sequential(
    [
        kerasLayers.Input(shape=(x.shape[1]), name="input"),
        kerasLayers.Dense(800, activation=wandb.config.layer2activation, name="layer1", use_bias=True),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        kerasLayers.Dropout(0.5),
        kerasLayers.Dense(400, activation=wandb.config.layer1activation, name="laye11", use_bias=True),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(300, activation=wandb.config.layer1activation, name="layer121", use_bias=False),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(200, activation=wandb.config.layer1activation, name="layer2341", use_bias=False),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(100, activation=wandb.config.layer1activation, name="laye21r1", use_bias=False),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(20, activation=wandb.config.layer1activation, name="laye4444r1"),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(400, activation=wandb.config.layer2activation, name="layer2"),  # , kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(25, activation="sigmoid", name="layer3", kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        # kerasLayers.Dense(25, activation="sigmoid", name="layer4", kernel_regularizer=kerasRegs.l1_l2(l1=1e-5, l2=1e-4)),
        # kerasLayers.Dropout(0.5),
        kerasLayers.Dense(y.shape[1], activation=wandb.config.layer3activation, name="output", use_bias=True)
    ]
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=wandb.config.learning_rate, epsilon=1   ),
              loss=keras.losses.MeanSquaredError(name='MSE_Loss'),
              metrics=[keras.metrics.MeanSquaredError(name='MSE_metric'),
                       keras.metrics.RootMeanSquaredError(name='RMS_metric'),
                       keras.metrics.MeanAbsoluteError(name='MAE_metric')])

history = model.fit(x_train,
                    y_train,
                    batch_size=wandb.config.batch_size,
                    epochs=wandb.config.epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback, WandbCallback()],
                    shuffle=True,
                    verbose=2,
                    )

model.save('./blaseball_playground/modelling/models/project_name.h5')

yp = model.predict(x)
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

error = y - yp
error_train = y_train - yp_train
error_test = y_test - yp_test

plt.hist(error, bins=50)
plt.show()

print('fuck')

a = 1
