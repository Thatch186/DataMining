import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, X=None, y=None, feature_names=None, label_name=None):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.label_name = label_name

    def set_X(self, X):
        self.X = X

    def set_y(self, y):
        self.y = y

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def set_label_name(self, label_name):
        self.label_name = label_name

    def read_csv(self, file_path, sep=','):
        df = pd.read_csv(file_path, sep=sep)
        self.X = df.drop(columns=[self.label_name])
        self.y = df[self.label_name]
        self.feature_names = self.X.columns.values

    def read_tsv(self, file_path, sep='\t'):
        df = pd.read_csv(file_path, sep=sep)
        self.X = df.drop(columns=[self.label_name])
        self.y = df[self.label_name]
        self.feature_names = self.X.columns.values

    def write_csv(self, file_path):
        df = pd.concat([self.X, self.y], axis=1)
        df.to_csv(file_path, index=False)

    def write_tsv(self, file_path):
        df = pd.concat([self.X, self.y], axis=1)
        df.to_csv(file_path, sep='\t', index=False)

    def describe(self):
        return self.X.describe()

    def count_null(self):
        return self.X.isnull().sum()

    def replace_null(self, value):
        self.X = self.X.fillna(value)

    def replace_null_with_most_common_value(self):
        for column in self.X.columns:
            mode_value = self.X[column].mode()[0]
            self.X[column] = self.X[column].fillna(mode_value)

    def replace_null_with_mean(self):
        for column in self.X.columns:
            mean_value = self.X[column].mean()
            self.X[column] = self.X[column].fillna(mean_value)

