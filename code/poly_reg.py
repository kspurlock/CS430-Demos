# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:12:01 2021

@author: kylei
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def train_test_overfitting(model, x, y, deg):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=4
    )

    poly_features = PolynomialFeatures(degree=deg)

    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.fit_transform(x_test)

    train_pred = model.predict(x_train_poly)
    test_pred = model.predict(x_test_poly)

    train_metric_dict = []
    test_metric_dict = []

    train_metric_dict.append(mean_squared_error(y_train, train_pred))
    train_metric_dict.append(mean_absolute_error(y_train, train_pred))
    train_metric_dict.append(r2_score(y_train, train_pred))

    test_metric_dict.append(mean_squared_error(y_test, test_pred))
    test_metric_dict.append(mean_absolute_error(y_test, test_pred))
    test_metric_dict.append(r2_score(y_train, train_pred))

    train_metric_dict = np.array(train_metric_dict).reshape(1, -1)
    test_metric_dict = np.array(test_metric_dict).reshape(1, -1)

    return_mat = np.column_stack((train_metric_dict, test_metric_dict))

    return return_mat

def find_median(matrix):
        medians = []
        for i in range(matrix.shape[1]):
            deg_med = np.median(matrix[:, [i]])
            medians.append(deg_med)

        return medians

if __name__ == "__main__":
    dataset = pd.read_csv(
        r"./data/covid_19_clean_complete.csv"
    )  # Read the training and test set data in
    dataset = dataset.loc[lambda a: pd.isna(a["Province/State"])]
    # countries = ['India', 'Russia', 'US', 'South Africa']#np.unique(x['Country/Region'])
    countries = np.unique(dataset["Country/Region"])
    # countries = ['Belize', 'Brazil','New Zealand', 'Serbia']

    # Initialize "empty" numpy arrays to hold aggregate metrics
    deg_1_agg = np.empty((1, 6))
    deg_2_agg = np.empty((1, 6))
    deg_3_agg = np.empty((1, 6))
    deg_4_agg = np.empty((1, 6))
    deg_5_agg = np.empty((1, 6))

    for i in countries:
        MSE_list = []
        MAE_list = []
        r2_list = []

        x = dataset.loc[lambda a: a["Country/Region"] == i]
        x = x.iloc[:, [4, 5, 8]]
        x["Date"] = pd.to_datetime(x["Date"])  # Convert the date column into datetime
        x["Date"] = x["Date"].map(dt.datetime.toordinal)  # Map datetime to ordinal

        x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)

        # x_plot = np.array(x['Date']) # Save the domain for use with plotting
        y = np.array(x["Confirmed"]).reshape(-1, 1)  # The target value
        x = x.iloc[:, [0, 2]]  # Select ordinal date and active cases column

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, random_state=4
        )
        x_train = np.sort(x_train, 0)
        x_test = np.sort(x_test, 0)
        x_test_plot = x_test[:, [0]]
        y_train = np.sort(y_train, 0)
        y_test = np.sort(y_test, 0)

        # Begin polynomial regression
        # Poly degree 1
        poly_feat = PolynomialFeatures(degree=1)
        x_train_poly = poly_feat.fit_transform(x_train)
        x_test_poly = poly_feat.fit_transform(x_test)

        regressor = LinearRegression()
        regressor.fit(x_train_poly, y_train)

        poly_prediction_1 = regressor.predict(x_test_poly)
        """Metrics"""
        poly_1_r2 = np.around(regressor.score(x_test_poly, y_test), 4)

        deg_1_metrics = train_test_overfitting(regressor, x, y, 1)

        # Poly degree 2
        poly_feat = PolynomialFeatures(degree=2)
        x_train_poly = poly_feat.fit_transform(x_train)
        x_test_poly = np.around(poly_feat.fit_transform(x_test), 4)

        regressor = LinearRegression()
        regressor.fit(x_train_poly, y_train)

        poly_prediction_2 = regressor.predict(x_test_poly)
        """Metrics"""
        poly_2_r2 = np.around(regressor.score(x_test_poly, y_test), 4)

        deg_2_metrics = train_test_overfitting(regressor, x, y, 2)

        # Poly degree 3
        poly_feat = PolynomialFeatures(degree=3)
        x_train_poly = poly_feat.fit_transform(x_train)
        x_test_poly = np.around(poly_feat.fit_transform(x_test), 4)

        regressor = LinearRegression()
        regressor.fit(x_train_poly, y_train)

        poly_prediction_3 = regressor.predict(x_test_poly)
        """Metrics"""
        poly_3_r2 = np.around(regressor.score(x_test_poly, y_test), 4)

        deg_3_metrics = train_test_overfitting(regressor, x, y, 3)

        # Poly degree 4
        poly_feat = PolynomialFeatures(degree=4)
        x_train_poly = poly_feat.fit_transform(x_train)
        x_test_poly = np.around(poly_feat.fit_transform(x_test), 4)

        regressor = LinearRegression()
        regressor.fit(x_train_poly, y_train)

        poly_prediction_4 = regressor.predict(x_test_poly)
        """Metrics"""
        poly_4_r2 = np.around(regressor.score(x_test_poly, y_test), 4)

        deg_4_metrics = train_test_overfitting(regressor, x, y, 4)

        # Poly degree 5
        poly_feat = PolynomialFeatures(degree=5)
        x_train_poly = poly_feat.fit_transform(x_train)
        x_test_poly = np.around(poly_feat.fit_transform(x_test), 4)

        regressor = LinearRegression()
        regressor.fit(x_train_poly, y_train)

        poly_prediction_5 = regressor.predict(x_test_poly)
        """Metrics"""
        poly_5_r2 = np.around(regressor.score(x_test_poly, y_test), 4)

        deg_5_metrics = train_test_overfitting(regressor, x, y, 5)

        plt.scatter(x_test_plot, y_test, color="grey")  # Plot the primary data points
        # Plot poly degree 1
        plt.plot(
            x_test_plot,
            poly_prediction_1,
            color="black",
            label="deg 1 " f"r\u00B2= {poly_1_r2}",
        )
        plt.plot(
            x_test_plot,
            poly_prediction_2,
            color="steelblue",
            label="deg 2 " f"r\u00B2= {poly_2_r2}",
        )
        # Plot poly degree 3
        plt.plot(
            x_test_plot,
            poly_prediction_3,
            color="green",
            label="deg 3 " f"r\u00B2= {poly_3_r2}",
        )
        plt.plot(
            x_test_plot,
            poly_prediction_4,
            color="darkorange",
            label="deg 4 " f"r\u00B2= {poly_4_r2}",
        )
        # Plot poly degree 5
        plt.plot(
            x_test_plot,
            poly_prediction_5,
            color="magenta",
            label="deg 5 " f"r\u00B2= {poly_5_r2}",
        )
        # plt.vlines(preventions, 0, max(y), color = 'blue', linestyles = 'dashed')
        plt.title("PolyReg - " + i)
        plt.xlabel("Date")
        plt.ylabel("Confirmed")
        plt.legend(loc="upper left", fontsize=12)
        plt.show()

        # Add all aggregate metrics for each country to ndarrays
        deg_1_agg = np.vstack((deg_1_agg, deg_1_metrics))
        deg_2_agg = np.vstack((deg_2_agg, deg_2_metrics))
        deg_3_agg = np.vstack((deg_3_agg, deg_3_metrics))
        deg_4_agg = np.vstack((deg_4_agg, deg_4_metrics))
        deg_5_agg = np.vstack((deg_5_agg, deg_5_metrics))

    # Empty numpy arrays initialize a random extreme value so delete that
    deg_1_agg = np.delete(deg_1_agg, 0, axis=0)
    deg_2_agg = np.delete(deg_2_agg, 0, axis=0)
    deg_3_agg = np.delete(deg_3_agg, 0, axis=0)
    deg_4_agg = np.delete(deg_4_agg, 0, axis=0)
    deg_5_agg = np.delete(deg_5_agg, 0, axis=0)

    # Find median for each degree
    deg_1_median = find_median(deg_1_agg)
    deg_2_median = find_median(deg_2_agg)
    deg_3_median = find_median(deg_3_agg)
    deg_4_median = find_median(deg_4_agg)
    deg_5_median = find_median(deg_5_agg)
