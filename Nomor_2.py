# -*- coding: utf-8 -*-
"""Nomor_2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dg1wgy1zl7m0wnKVu7xe3NgJ_RivqzoK
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
import joblib

class data_set:
  # read data
  def __init__(self, file_name):
    self.file_name = file_name
    self.file = pd.read_csv(file_name)
    self.input = None
    self.output = None

  #menampilkan beberapa informasi penting dalam dataset
  def information(self):
        print("*"*100)
        print('Data Information: ')
        print(self.file.info())
        print()
        print('Dataset Statistics : ')
        print(self.file.describe())
        print()
        print('Null values : ')
        print(self.file.isna().sum())
        print("*"*100)

  # split dataset 80% train dan 20% test
  def split(self, column):
    self.input = self.file.drop(columns = [column])
    self.output = self.file[column]
    x_train, x_test, y_train, y_test = train_test_split(self.input, self.output, test_size = 0.2, random_state = 42)
    return [x_train,y_train],[x_test,y_test]

class Handling_dataset:
  def __init__(self, train_data, test_data):
    self.x_train, self.y_train = train_data
    self.x_test, self.y_test = test_data
    self.model = None

  #handle missing value
  def missing_value(self, columns, value):
    self.x_train[columns] = self.x_train[columns].fillna(value)
    self.x_test[columns] = self.x_test[columns].fillna(value)

  # feature encode
  def feature_encoding(self, label_encoding):
    self.x_train = self.x_train.replace(label_encoding)
    self.x_test = self.x_test.replace(label_encoding)

  # one hot encoding
  def one_hot_encode(self, columns1):
    train_encoded = OneHotEncoder()
    x_encode = self.x_train[columns1]
    test_data = pd.DataFrame(train_encoded.transform(x_encode_2).toarray(), columns = train_encoded.get_feature_names_out())
    self.x_train = self.x_train.reset_index()
    self.x_train =pd.concat([self.x_train, train_data], axis=1)

    x_encode_2 = self.x_test[columns1]
    train_data = pd.DataFrame(train_encoded.fit_transform(x_encode).toarray(), columns = train_encoded.get_feature_names_out())
    self.x_test = self.x_test.reset_index()
    self.x_test = pd.concat([self.x_test, test_data], axis=1)

  def filter_columns(self, columns2):
    self.x_train = self.x_train[columns2]
    self.x_test = self.x_test[columns2]

  def modeling(self):
    self.model = joblib.load('XGBoost.pkl')

  def fitting(self):
    self.model = self.model.fit(self.x_train, self.y_train)

  def predict(self):
    self.y_predict = self.model.predict(self.x_test)

  def report(self, target_names):
    print(classification_report(self.y_test, self.y_predict, target_names = target_names))

df = data_set('data_A.csv')
df.information()

train_data, test_data = df.split('churn')

model = Handling_dataset(train_data, test_data)

encode = {"Gender": {"Male":1,"Female" :0}}
model.feature_encoding(encode)

model.missing_value('CreditScore',657)

filtering = ['CreditScore', 'Gender', 'Age',
                     'Tenure', 'Balance','NumOfProducts',
                     'HasCrCard', 'IsActiveMember',
                     'EstimatedSalary']
model.filter_columns(filtering)

model.modeling()

model.fitting()

model.predict()

model.report(['1','0'])