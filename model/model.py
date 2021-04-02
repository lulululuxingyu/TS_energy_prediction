'''
Created on 02 Apr 2021
Create Model

@author: Xingyu Lu
'''
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


class dataset():
    def __init__(self, df, train_window, datafields):
        num_test = int(math.sqrt(df.shape[0]))
        num_train = df.shape[0] - num_test
        self.train_window = train_window
        self.scalers = {f: MinMaxScaler(feature_range=(-1,1)) for f in datafields}
        self.train_normalized, self.test_labels = self.load_train_data_normalized(df, datafields, num_train)
        #print(self.train_normalized, self.test_labels)
        
        
    def load_train_data_normalized(self, df, fields, num_train):
        train_normalized = []
        test = []
        for f in fields:
            f_data = df[f].values.reshape((-1, 1))
            train_data = f_data[:num_train]
            scaler = self.scalers[f]
            train_normalized.append(scaler.fit_transform(train_data))
            test.append(f_data[num_train:])
        return np.hstack(train_normalized), np.hstack(test)
    
    def load_train_sequence(self):
        '''
        return [seq, label]
        '''
        inout_seq = []
        L = len(self.train_normalized)
        tw = self.train_window
        for i in range(L-tw):
            train_seq = self.train_normalized[i:i+tw]
            train_label = self.train_normalized[i+tw:i+tw+1]
            inout_seq.append((train_seq ,torch.FloatTensor(train_label)))
        return inout_seq
    
    def load_test_inputs(self):
        test_inputs = self.train_normalized[-self.train_window:]
        #print(test_inputs.T.tolist()[0])
        #print(self.test_labels)
        return test_inputs.T.tolist()[0], self.test_labels



class LSTM_double(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(torch.FloatTensor(input_seq).view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTM_single(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        #self.hidden_cell
        lstm_out, _ = self.lstm(torch.FloatTensor(input_seq).view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

