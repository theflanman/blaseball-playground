import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.ensemble
import wandb
import wandb.sklearn

import blaseball_playground.config_matplotlib
import matplotlib.pyplot as plt

from blaseball_playground.modelling.common import get_data


def main():
    project_name = "skl_regressions"
    wandb.init(project=project_name)
    wandb.config.regressor = 'SVR'
    wandb.config.dataset = 'forbidden'

    x1, y = get_data('forbidden')
    yc = y.copy()
    x2, _ = get_data('star')
    x = np.hstack((x1, x2))
    x /= x.std(axis=0)
    x -= x.mean(axis=0)
    pca = sklearn.decomposition.PCA(10)
    pca.fit(x, y)
    x = pca.transform(x)
    y = 1*(np.diff(y)[:, 0] > 0)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=0)

    regressors = {'LinearRegression': sklearn.linear_model.LinearRegression,
                  'Ridge': sklearn.linear_model.Ridge,
                  'ElasticNet': sklearn.linear_model.ElasticNet,
                  'RandomForest': sklearn.ensemble.RandomForestRegressor,
                  'SVR': sklearn.svm.SVR,
                  }

    # pca = sklearn.decomposition.PCA(16)
    #
    # pca.fit(X_train)
    #
    # x, X_train, X_test = tuple(pca.transform(var) for var in [x, X_train, X_test])

    # model = sklearn.pipeline.Pipeline([
    #     # ('PCA', sklearn.decomposition.PCA(50)),
    #     ('regressor', regressors[wandb.config.regressor](kernel=)),  #
    # ])
    model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    yp = model.predict(x)
    yp_train = model.predict(X_train)
    yp_test = model.predict(X_test)


    error = y - yp
    error_train = y_train - yp_train
    error_test = y_test - yp_test

    # print(np.sqrt(((error**2).sum(axis=1)).mean()))

    wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test, model_name=wandb.config.regressor)
    #
    # wandb.log({'error_rms': np.sqrt((error_train**2).sum(axis=1)), 'validation_error_rms': np.sqrt((error_test**2).sum(axis=1))})
    # wandb.summary({'error_rms': np.sqrt((error_train**2).sum(axis=1)).mean(), 'validation_error_rms': np.sqrt((error_test**2).sum(axis=1)).mean()})

    pass


if __name__ == '__main__':
    main()
