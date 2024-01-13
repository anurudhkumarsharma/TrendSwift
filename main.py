from datetime import datetime
import os
import sys
import time
import requests
import numpy as np
import pandas as pd

pd.options.plotting.backend = "plotly"

# pip install yfinance
# conda install -c postelrich yahoo-finance
# pip install nsetools
# pip install nsepython
# pip install mpld3
# pip install numpy==1.23.4

import yfinance as yf
import plotly.graph_objs as go

from PyQt5 import QtCore, QtWebEngineWidgets
from PyQt5.QtCore import (QTimer, QAbstractTableModel, QUrl)
from PyQt5.QtCore import pyqtSlot, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import matplotlib.pyplot as plt, mpld3

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Forecasting:
    def __init__(self):
        super(Forecasting, self).__init__()
        self.df = None

    def Do_Forecast(self):

        self.df = pd.read_csv('data.csv')
        # setting index as date
        self.df['Date'] = pd.to_datetime(self.df.Date, format='%Y-%m-%d')
        self.df.index = self.df['Date']
        original_fig = plt.figure(figsize=(16, 8))
        plt.plot(self.df['Close'])
        plt.ylabel('Close Price history (INR)')
        plt.xlabel("Year")
        # plt.title("Reliance-Industries-Limited \n[https://www.quandl.com/data/NSE/RELIANCE-Reliance-Industries-Limited]")
        mpld3.save_html(original_fig, "original_chart.html")

        # creating dataframe by reversing the order of the timeseries with latest year at the bottom
        data = self.df.sort_index(ascending=True, axis=0)
        indx = self.df.index.get_loc('2022-01-03')

        new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Close'][i] = data['Close'][i]

        # setting index
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        # creating train and test sets
        dataset = new_data.values

        # Now find the index of the date '2022-01-01'
        indx_val_start = new_data.index.get_loc('2022-01-03')

        # find the index of the last day year 2022
        indx_val_end = new_data.index.get_loc('2022-12-30')

        # Let us now divide the Whole Dataset into training and validation sets
        # according to the indexes of the years found in the previous steps

        train = dataset[0:indx_val_start, :]
        valid = dataset[indx_val_start:indx_val_end, :]

        # Feature Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # converting dataset into x_train and y_train

        x_train, y_train = [], []

        for i in range(60, len(train)):
            x_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        print('x_train:', x_train.shape, '\ny_train', y_train.shape)

        # create and fit the LSTM network
        # return_sequences: Boolean. Whether to return the last output.
        # in the output sequence, or the full sequence. Default: False.
        # https://keras.io/api/models/model_training_apis/#fit-method

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        History = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

        # predicting 246 values, using past 60 from the train data

        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        predicted_closing_price = scaler.inverse_transform(closing_price)

        # for plotting

        train = new_data[1:indx_val_start]
        valid = new_data[indx_val_start:indx_val_end]

        # Plot the data
        valid = valid.assign(Predictions=0)
        valid = valid.assign(Predictions=predicted_closing_price)

        # valid["Predictions"] = predicted_closing_price

        # plt.plot(train['Close'])
        # plt.plot(valid[['Close','Predictions']])

        # Visualize the data
        predicted_fig = plt.figure(figsize=(16, 8))
        plt.title('Stock Market Closing Price Prediction using LSTM')
        plt.xlabel('Year', fontsize=18)
        plt.ylabel('Close Price INR (₹)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        mpld3.save_html(predicted_fig, "predicted_chart.html")


class Forecasting_days():
    def __init__(self) -> None:
        super(Forecasting_days, self).__init__()
        self.df = None

    def Do_Days_Forecast(self, days_to_predict):

        self.df = pd.read_csv('data.csv')
        original_fig = plt.figure(figsize=(15, 8))
        plt.plot(self.df['Close'])
        plt.ylabel('Close Price history (INR)')
        plt.xlabel("Days")
        # plt.xticks(self.df['Date'].to_list())
        # plt.yticks(self.df['Close'].to_list())
        mpld3.save_html(original_fig, "original_chart.html")        

        y = self.df['Close'].fillna(method='ffill')
        y = y.values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)

        n_lookback = 60  # length of input sequences (lookback period)
        # n_forecast = 30  # length of output sequences (forecast period)
        n_forecast = int(days_to_predict)  # length of output sequences (forecast period)

        X = []
        Y = []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        X = np.array(X)
        Y = np.array(Y)

        # Fit Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(n_forecast))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=100, batch_size=32, verbose=1)

        # Generate Forecast
        X_ = y[- n_lookback:]  # last available input sequence
        X_ = X_.reshape(1, n_lookback, 1)

        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)

        # organize the results in a data frame

        df_past = self.df[['Close']].reset_index()
        df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
        # df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Date'] = pd.to_datetime(self.df['Date'])
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan

        results = pd.concat([df_past, df_future], ignore_index=True)
        x_axis_date_time = pd.to_datetime(results['Date'])
        y_data = results[['Actual', 'Forecast']]
        y_data = y_data.set_index(x_axis_date_time)
        fig = y_data.plot(kind='line')
        fig.write_html("predicted_chart.html")
        
        results.to_csv('forecast_results.csv')

        return df_past, df_future, results


class OptionChainAnalysis():
    def __init__(self) -> None:
        super(OptionChainAnalysis, self).__init__()

        self.url_oc = "https://www.nseindia.com/option-chain"
        self.url = "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY"
        self.headers = {"accept-encoding": "gzip, deflate, br",
                        "accept-language": "en-US,en;q=0.9,hi;q=0.8,vi;q=0.7",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
                        }

    def Fetch_Expiry_Strike_price(self, symbol_name):
        self.url = "https://www.nseindia.com/api/option-chain-indices?symbol=" + str(symbol_name)
        session = requests.Session()
        data = session.get(self.url, headers=self.headers).json()["records"]["data"]
        ocdata = []

        for i in data:
            for j, k in i.items():
                if j == "CE" or j == "PE":
                    info = k
                    info["intrumentType"] = j
                    ocdata.append(info)
        df = pd.DataFrame(ocdata)
        session.close()

        # expiry_date_list = df['expiryDate'].unique().tolist()
        # strike_price_list = df['strikePrice'].unique().tolist()

        expiry_date_list = df['expiryDate'].sort_values().unique().tolist()
        strike_price_list = list(map(str, df['strikePrice'].sort_values().unique().tolist()))

        return expiry_date_list, strike_price_list

    def Fetch_NSEOptionData(self, symbol_name):
        self.url = "https://www.nseindia.com/api/option-chain-indices?symbol=" + str(symbol_name)

        session = requests.Session()
        request = session.get(self.url, headers=self.headers)
        cookies = dict(request.cookies)

        response = session.get(self.url, headers=self.headers, cookies=cookies).json()
        rawdata = pd.DataFrame(response)
        rawop = pd.DataFrame(rawdata['filtered']['data']).fillna(0)
        data = []

        for i in range(0, len(rawop)):
            calloi = callcoi = cltp = putoi = putcoi = pltp = 0
            stp = rawop['strikePrice'][i]
            if (rawop['CE'][i] == 0):
                calloi = callcoi = 0
            else:
                calloi = rawop['CE'][i]['openInterest']
                callcoi = rawop['CE'][i]['changeinOpenInterest']
                cltp = rawop['CE'][i]['lastPrice']
            if (rawop['PE'][i] == 0):
                putoi = putcoi = 0
            else:
                putoi = rawop['PE'][i]['openInterest']
                putcoi = rawop['PE'][i]['changeinOpenInterest']
                pltp = rawop['PE'][i]['lastPrice']
            opdata = {
                'CALL OI': calloi, 'CALL CHNG OI': callcoi, 'CALL LTP': cltp, 'STRIKE PRICE': stp,
                'PUT OI': putoi, 'PUT CHNG OI': putcoi, 'PUT LTP': pltp
            }

            data.append(opdata)
        optionchain = pd.DataFrame(data)
        return optionchain


""" -------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------------------------------- """


class StockApp(QMainWindow):
    def __init__(self):
        super(StockApp, self).__init__()

        loadUi("StockApp.ui", self)

        symbol_df = pd.read_csv('ind_nifty50list.csv')
        symbol_list = symbol_df['Symbol'].to_list()
        self.ticker_comboBox.addItems(symbol_list)

        self.forecasting_obj = Forecasting()
        self.forcasting_days = Forecasting_days()
        self.option_chain_data_Obj = OptionChainAnalysis()

        self.searchTicker_pushButton.clicked.connect(self.search_TickerData_Function)
        self.forecast_pushButton.clicked.connect(self.forecastStock_Function)
        
    @pyqtSlot()
    def search_TickerData_Function(self):
        symbol_name = self.ticker_comboBox.currentText()

        if symbol_name:
            yf_ticker_name = symbol_name + '.NS'
            print(yf_ticker_name)

            # data = yf.download(tickers=yf_ticker_name, start="2013-01-01" , end="2023-01-01", period='10y', interval='1d')

            if self.year_checkBox.isChecked():
                data = yf.download(tickers=yf_ticker_name, period='10y', interval='1d')
            else:
                data = yf.download(tickers=yf_ticker_name, period='1y')

            if os.path.isfile('data.csv'):
                os.remove('data.csv')
                print("Old Data Deleted !")

            time.sleep(1)
            data.to_csv('data.csv')

            data_table = data
            data_table = data_table.reset_index()

            model_data = self.pandasModel(data_table)
            self.data_tableView.setModel(model_data)

    @pyqtSlot()
    def forecastStock_Function(self):

        if self.year_checkBox.isChecked():
            self.forecasting_obj.Do_Forecast()
        else:
            if self.days_spinBox.value():
                df_past, df_future, results = self.forcasting_days.Do_Days_Forecast(self.days_spinBox.value())

                model_data_past = self.pandasModel(df_past)
                self.past_values_tableView.setModel(model_data_past)

                model_data_future = self.pandasModel(df_future)
                self.future_values_tableView.setModel(model_data_future)

                model_data_results = self.pandasModel(results)
                self.forecast_values_tableView.setModel(model_data_results)
            else:
                QMessageBox.warning(self, "Error", "Please enter Days to predict !")

        html_abs_path = os.path.abspath("original_chart.html")
        url = QUrl.fromLocalFile(html_abs_path)
        self.chart_webEngineView.load(url)
        self.chart_webEngineView.show()

        html_abs_path = os.path.abspath("predicted_chart.html")
        url = QUrl.fromLocalFile(html_abs_path)
        self.forecast_webEngineView.load(url)
        self.forecast_webEngineView.show()

    @pyqtSlot()
    def Fetch_and_Display_OC_Data_Function(self):
        symbol_name = self.index_comboBox.currentText()

        df = self.option_chain_data_Obj.Fetch_NSEOptionData(symbol_name)
        model_data = self.pandasModel(df)
        self.optionChain_tableView.setModel(model_data)

    class pandasModel(QAbstractTableModel):

        def __init__(self, data):
            QAbstractTableModel.__init__(self)
            self._data = data

        def rowCount(self, parent=None):
            return self._data.shape[0]

        def columnCount(self, parent=None):
            return self._data.shape[1]

        def data(self, index, role=QtCore.Qt.DisplayRole):
            if index.isValid():
                if role == QtCore.Qt.DisplayRole:
                    return str(self._data.iloc[index.row(), index.column()])
                return None

        def headerData(self, col, orientation, role):
            if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
                return self._data.columns[col]
            return None


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = StockApp()
    window.show()
    sys.exit(app.exec_())
