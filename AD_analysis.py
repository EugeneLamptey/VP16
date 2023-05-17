import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


class AD_analysis():
    def __init__(self, X_train):
        # Setting the variables and threshold
        self.X_train = scaler.fit_transform(X_train)
        self.h_thres = 3 * (len(X_train.columns) + 1) / len(X_train)
        self.S_thres = 3

        self.mean = np.mean(self.X_train, axis=1)
        self.std = np.std(self.X_train, axis=1)

    def standardized_similarity(self):
        s_new = (self.mean + 1.28 * self.std)
        self.s_new = s_new
        return self.s_new

    def leverage_values(self):
        # Leverage values are calculated from the diagonal of the hat matrix
        hat_matrix = np.dot(self.X_train, np.linalg.inv(np.dot(self.X_train.T, self.X_train))).dot(self.X_train.T)
        leverage_values = np.diagonal(hat_matrix)
        self.lev_values = leverage_values
        return self.lev_values

    def visualize_AD(self):
        self.s_new = self.standardized_similarity()
        self.lev_values = self.leverage_values()


        plt.scatter(self.lev_values, self.s_new)
        plt.axhline(y=self.S_thres, color='r', linestyle='-', linewidth = 2.0)  #plots a horizontal line to denote the max and min y threshold
        plt.axhline(y = -self.S_thres, color = "r", linestyle = "-", linewidth = 2.0) #plots another horizontal line at threshold -3
        plt.axvline(x = self.h_thres, color = "r", linestyle = "-", linewidth = 2.0) #plots a vertical line to denote the H_threshold
        

        plt.xlabel("Leverage Values")
        plt.ylabel("Residuals")
        plt.title("Applicability Domain")
        plt.show()
  
