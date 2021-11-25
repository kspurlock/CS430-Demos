# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:33:35 2021

@author: kylei
"""
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_test_overfitting(model, x, y, is_penalized=False, is_elastic=False):
    """Method for training a model and compiling metrics"""
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=4
    )

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_metric_dict = []
    test_metric_dict = []

    train_metric_dict.append(mean_squared_error(y_train, train_pred))
    train_metric_dict.append(mean_absolute_error(y_train, train_pred))
    train_metric_dict.append(model.score(x_train, y_train))

    test_metric_dict.append(mean_squared_error(y_test, test_pred))
    test_metric_dict.append(mean_absolute_error(y_test, test_pred))
    test_metric_dict.append(model.score(x_test, y_test))

    train_metric_dict = np.array(train_metric_dict).reshape(1, -1)
    test_metric_dict = np.array(test_metric_dict).reshape(1, -1)

    return_mat = np.column_stack((train_metric_dict, test_metric_dict))

    return return_mat


def find_meds(matrix):
    ret_list = []

    for i in range(matrix.shape[1]):
        ret_list.append(np.median(matrix[:, [i]]))

    return ret_list

if __name__ == "__main__":
    # Intialize aggregated arrays for each regularization technique
    agg_LR = np.empty((1, 6))
    agg_MLR = np.empty((1, 6))
    agg_RR = np.empty((1, 6))
    agg_EN = np.empty((1, 6))

    dataset_orig = pd.read_csv(
        r"./data/covid_19_clean_complete.csv"
    )  # Read the training and test set data in
    countries = np.unique(dataset_orig["Country/Region"].values)
    countries = ["Brazil"]  # Just look at this one country
    for i in countries:
        country = i
        dataset = dataset_orig.loc[dataset_orig["Country/Region"] == country]

        dataset = dataset.drop(["Province/State", "Country/Region"], axis=1)
        dataset["Date"] = pd.to_datetime(
            dataset["Date"]
        )  # Convert the date column into datetime
        dataset["Date"] = dataset["Date"].map(
            dt.datetime.toordinal
        )  # Map datetime to ordinal
        CT = ColumnTransformer(
            [("encoder", OneHotEncoder(), [7])], remainder="passthrough"
        )
        df = pd.DataFrame(
            CT.fit_transform(dataset),
            columns=[
                "WHO Region",
                "Lat",
                "Long",
                "Date",
                "Confirmed",
                "Deaths",
                "Recovered",
                "Active",
            ],
        )
        df = df.drop(["WHO Region", "Lat", "Long"], axis=1)
        df = df.iloc[:, [0, 1, 4]]

        """
        # Can be used to show feature correlation per country

        cols = np.array(df.columns)
        
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cm,
                        cbar=True,
                        annot=True,
                        square=True,
                        cmap = 'Blues',
                        fmt='.2f',
                        annot_kws={'size': 12},
                        yticklabels=cols,
                        xticklabels=cols)
        plt.title('"covid_complete_clean.csv" Correlation')
        plt.show()
        """

        data_scaled = pd.DataFrame(
            StandardScaler().fit_transform(df), columns=df.columns
        )

        # For easily modulating which x and y axis to plot
        inpy = 1
        inpx = 0

        xlabel = df.columns[inpx]
        ylabel = df.columns[inpy]

        x = data_scaled.loc[:, df.columns != inpy].values
        y = data_scaled.iloc[:, inpy].values

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0
        )

        x_train = np.sort(x_train, 0)
        x_test = np.sort(x_test, 0)
        y_train = np.sort(y_train, 0)
        y_test = np.sort(y_test, 0)

        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        coef = regressor.coef_
        intercept = regressor.intercept_
        SLR_r2 = np.around(regressor.score(x_test, y_test), 4)

        # Grid search on L1 and L2 penalized models

        L_param_grid = [
            {
                "alpha": np.linspace(0.15, 5, num=100),
                "tol": np.linspace(1e-4, 90, num=100),
            }
        ]

        R_param_grid = [
            {
                "alpha": np.linspace(0.01, 10, num=100),
                "tol": np.linspace(1e-4, 90, num=100),
            }
        ]

        en_param_grid = [
            {
                "alpha": np.linspace(0.1, 1, num=10),
                "l1_ratio": np.linspace(0.01, 1, num=10),
                "tol": np.linspace(1e-4, 90, num=100),
            }
        ]
        score_metric = "neg_mean_squared_error"

        clf_LR = RandomizedSearchCV(Lasso(), L_param_grid, scoring=score_metric)
        clf_LR.fit(x_train, y_train)
        LR = clf_LR.best_estimator_
        print(
            "Score: {s}, Alpha: {a}".format(
                s=LR.score(x_test, y_test), a=LR.get_params()["alpha"]
            )
        )

        clf_RR = RandomizedSearchCV(Ridge(), R_param_grid, scoring=score_metric)
        clf_RR.fit(x_train, y_train)
        RR = clf_RR.best_estimator_
        print(
            "Score: {s}, Alpha: {a}".format(
                s=RR.score(x_test, y_test), a=RR.get_params()["alpha"]
            )
        )

        clf_EN = RandomizedSearchCV(ElasticNet(), en_param_grid, scoring=score_metric)
        clf_EN.fit(x_train, y_train)
        EN = clf_EN.best_estimator_
        print(
            "Score: {s}, Alpha: {a}, L1 Ratio: {l}".format(
                s=EN.score(x_test, y_test),
                a=EN.get_params()["alpha"],
                l=EN.get_params()["l1_ratio"],
            )
        )

        LR_R2 = np.around(LR.score(x_test, y_test), 4)
        RR_R2 = np.around(RR.score(x_test, y_test), 4)
        EN_R2 = np.around(EN.score(x_test, y_test), 4)

        MLR_metrics = train_test_overfitting(regressor, x, y)
        LR_metrics = train_test_overfitting(LR, x, y)
        RR_metrics = train_test_overfitting(RR, x, y)
        EN_metrics = train_test_overfitting(EN, x, y)

        agg_MLR = np.vstack((agg_MLR, MLR_metrics))
        agg_LR = np.vstack((agg_LR, LR_metrics))
        agg_RR = np.vstack((agg_RR, RR_metrics))
        agg_EN = np.vstack((agg_EN, EN_metrics))

        plt.scatter(x[:, inpx], y, color="grey")
        plt.plot(
            x_test[:, inpx],
            regressor.predict(x_test),
            color="black",
            label="MLR " f"r\u00B2= {SLR_r2}",
        )
        plt.plot(
            x_test[:, inpx],
            LR.predict(x_test),
            color="cyan",
            label="LASSO " f"r\u00B2= {LR_R2}",
        )
        plt.plot(
            x_test[:, inpx],
            RR.predict(x_test),
            color="orange",
            label="Ridge " f"r\u00B2= {RR_R2}",
        )
        plt.plot(
            x_test[:, inpx],
            EN.predict(x_test),
            color="magenta",
            label="ElasticNet " f"r\u00B2 = {EN_R2}",
        )

        plt.legend(loc="best", fontsize="xx-small")
        # plt.scatter(x_test, regressor.predict(x_test), color = 'grey')
        plt.title(country + " - " + xlabel + " vs. " + ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    # Format is train MSE, train MAE, train R2, test MSE, test MAE, test R2

    agg_MLR = np.delete(agg_MLR, 0, axis=0)
    agg_LR = np.delete(agg_LR, 0, axis=0)
    agg_RR = np.delete(agg_RR, 0, axis=0)
    agg_EN = np.delete(agg_EN, 0, axis=0)

    MLR_meds = find_meds(agg_MLR)
    LR_meds = find_meds(agg_LR)
    RR_meds = find_meds(agg_RR)
    EN_meds = find_meds(agg_EN)
