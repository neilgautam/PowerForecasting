####code to  upload the data and resample the data (hour wise)
from sklearn.externals import joblib
from keras.layers import Dense ,LSTM
from keras.models import Sequential 
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
import matplotlib.pyplot as plt

import r2
import mane

import keras.backend as K

def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SSa_res/(SS_tot + K.epsilon()))

### importing the data files
x_year = joblib.load("x_year.pkl")
y_year = joblib.load("y_year_values.pkl")

###slicing the tiestamp from x_year
timestamps = x_year[:,0:1]

#preparing data file for rnn
rnn_data  = np.c_[timestamps,y_year[:,0:1]]

'''
file = open("channel_5.dat")
data = file.readlines()
rnn_data=[]
for i in data:
    temp = i.split(" ")
    rnn_data.append([int(x) for x in temp])
'''
###changing the data file to dataframe 
rnn_data = pd.DataFrame(rnn_data)
rnn_data = rnn_data.set_index([0])
rnn_data.index = pd.to_datetime(rnn_data.index,unit = "s")

#resampling the data hourwise
resampler_hour = rnn_data.resample("H")
hour_resampled_data = resampler_hour.sum()

#saving the resampled file in the pickle format 
#joblib.dump(hour_resampled_data,"day_resampled_data.pkl")
#changing the dataframe to array
final_hour_array = np.asarray(hour_resampled_data)

errors_saver = {}

#selecting the size of
print("Enter the fraction of data to be sent to training :") 
training_fraction = float(input())
length = len(final_hour_array)
train_size = int(training_fraction*length)

for i in range(final_hour_array.shape[1]):
    
    copy = final_hour_array[:,i:i+1]
    sc = MinMaxScaler()
    copy = sc.fit_transform(copy)
    
    #now we are going to divide into training and test set
    x_train = copy[:train_size,0:1]
    y_train = copy[:train_size,0:1]
    #x_train = x_train.reshape((-1,1,1))
    train_x = []
    train_y = []
    print("enter n steps : ")
    n_steps = int(input())
    list_i = 0
    print("Entering loop")
    print(x_train.shape)
    print(y_train.shape)
    k=0
    while k<x_train.shape[0]-n_steps-1:
        train = x_train[k:k+n_steps,0]
        train_x.append([float(x) for x in train])
        train_y.append(y_train[k+n_steps,0])
        k= k+1
    
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)    
    train_x = train_x.reshape((-1,n_steps,1))
    train_y = train_y.reshape((-1,1,1))
    
    x_test = copy[train_size:,:]
    y_test = copy[train_size:,:]
    #x_test = x_test.reshape((-1,1,1))
    test_x = []
    test_y = []
    k=0
    while k<x_test.shape[0]-n_steps-1:
        test= x_test[k:k+n_steps,0]
        test_x.append([x for x in test])
        test_y.append(y_test[k+n_steps,:])
        k = k+1
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)    
    
    test_x = test_x.reshape((-1,n_steps,1))
    test_y = test_y.reshape((-1,1,1))
    
    '''    
    regressor = Sequential()
    regressor.add(LSTM(units = 10,input_shape= (n_steps,1))
    regressor.add(LeakyReLU(alpha = 0.05))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = "adam",loss = "mean_squared_error",metrics=["accuracy"])
    history = regressor.fit(train_x,train_y,epochs = 100 ,batch_size = 24,validation_split = 0.25,shuffle = False)
    '''
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps,1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(1))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss=mane.mane_loss, optimizer='adam',metrics = ["accuracy", r2])
	# fit network
    history = model.fit(train_x, train_y, epochs=10, batch_size=5,validation_split = 0.1)
    
    
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    y_pred = model.predict(test_x)
    y_pred = y_pred.reshape((-1,1))
    test_y = test_y.reshape((-1,1))
    
    mse = mean_squared_error(test_y,y_pred)
    rms = sqrt(mse)
    mae = mean_absolute_error(test_y,y_pred)
    print("mean_squared_error : ",mse)
    print("root_mean_squared_error : ",rms)
    print("mean_absolute_error : ",mae)
    errors_saver["errors_equipment{}".format(i+1)] = {"mean_squared_error":mse,"mean_absolute_error":mae,"root_mean_squared_error":rms}
    
    y_pred = sc.inverse_transform(y_pred)
    y_test = sc.inverse_transform(y_test)


    ik=0
    while ik<len(y_test):
        plt.scatter([x for x in range(1,25)],y_pred[ik:ik+24,:],color  = "cyan",label = "predicted_day")
        plt.scatter([x for x in range(1,25)],y_test[ik:ik+24,:],color = "blue",label = "real_values")
        plt.legend()
        plt.xlabel("day")
        plt.ylabel("power_consumption")
        plt.show()
        ik = ik+24
    print("Enter 1 if you want to process next equipment and 0 if not :")

    decision = int(input())
    
    if decision==1:
        pass
    else:
        break
    
    
            
