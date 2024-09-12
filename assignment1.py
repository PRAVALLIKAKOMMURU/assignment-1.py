import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
class TRAINING:
    def __init__(self,location):
        try:
            self.df = pd.read_csv(location,encoding='latin1')
            self.df = self.df.drop(['Customer Name','Customer e-mail','Country'],axis=1)
            self.X = self.df.iloc[: , :-1] # independent
            self.y = self.df.iloc[: , -1] # dependent
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')
    def data_training(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train,self.y_train) # training
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')
    def train_performance(self):
        try:
            self.y_train_pred = self.reg.predict(self.X_train)
            print(f' Train Accuracy : -> {r2_score(self.y_train,self.y_train_pred)}')
            print(f'Train Loss : {mean_squared_error(self.y_train,self.y_train_pred)}')
            print(f'train loss using mean absolute {mean_absolute_error(self.y_train,self.y_train_pred)}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')
    def testing(self):
        try:
            self.y_test_pred = self.reg.predict(self.X_test)
            print(f' Test Accuracy : -> {r2_score(self.y_test, self.y_test_pred)}')
            print(f'Test Loss : {mean_squared_error(self.y_test,self.y_test_pred)}')
            print(f'test loss using mean absolute {mean_absolute_error(self.y_test,self.y_test_pred)}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')

if __name__ == "__main__":
    try:
        obj = TRAINING('C:\\Users\\HP\\Downloads\\ML\\MLR_Project\\Car_Purchasing_Data.csv') # constructor will be called
        obj.data_training()
        obj.train_performance()
        obj.testing()
    except Exception as e:
        error_type,error_msg,err_line = sys.exc_info()
        print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')