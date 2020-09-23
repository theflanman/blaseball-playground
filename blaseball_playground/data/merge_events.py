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


@nb.njit(nb.boolean(nb.float64[:], nb.float64[:], nb.uint64))
def arr_compare(a, b, count):
    for i in range(count):
        if a[i] != b[i]:
            return False
    return True


@nb.njit()
def _combine_events(i, row, y_unique, x_full, y_full, indices):
    new_y = np.zeros(y_full.shape[1])
    count = 0
    for j in range(x_full.shape[0]):
        if indices[j]:
            if arr_compare(row, x_full[j], x_full.shape[1]):
                new_y += y_full[j, :]
                count += 1
                indices[j] = False
    y_unique[i, :] = new_y/count


def combine_events(events, dep_vars, ind_vars):
    x_full = events[ind_vars].to_numpy()
    y_full = events[dep_vars].to_numpy()
    x_unique = np.unique(x_full, axis=0)
    y_unique = np.zeros((x_unique.shape[0], y_full.shape[1]))
    indices = np.ones(y_full.shape[0], np.int8)
    for i, row in tqdm.tqdm(enumerate(x_unique), total=x_unique.shape[0]):
        _combine_events(i, row, y_unique, x_full, y_full, indices)
    y_unique /= y_unique.sum(axis=1)[:, np.newaxis]
    return pd.DataFrame(x_unique, columns=ind_vars), pd.DataFrame(y_unique, columns=dep_vars)


def main():
    events = pd.read_hdf('./blaseball_playground/data/files/events/events.hdf5', 'events')
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

    x, y = combine_events(events, dep_vars, ind_vars)

    merged_events = pd.concat((x, y), axis='columns')

    merged_events.to_hdf('./blaseball_playground/data/files/events/events.hdf5', 'merged_events')


if __name__ == '__main__':
    main()
