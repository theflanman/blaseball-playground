import numba as nb
import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.multiclass
import sklearn.model_selection
import sklearn.pipeline
import sklearn.ensemble
import tqdm
import wandb
import wandb.sklearn

import blaseball_playground.config_matplotlib
import matplotlib.pyplot as plt


def main():
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

    pipeline = sklearn.pipeline.Pipeline([
        ('classifier', sklearn.linear_model.LinearRegression())
    ])

    parameters = {
        'classifier': (
            sklearn.linear_model.LinearRegression(),
            sklearn.linear_model.Ridge(),
        ),
    }

    grid_search = sklearn.model_selection.GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    grid_search.fit(x, y)

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print(f'    {param_name}: {best_parameters[param_name]}')

    pass


if __name__ == '__main__':
    main()
