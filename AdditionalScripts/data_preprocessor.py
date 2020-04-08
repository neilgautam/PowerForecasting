#importing numpy , pandas and matplotlib to work on the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

#the function for processing the data to turn it into a single variable which contain all instances
def data_preprocessing(dataset_m):
    
    dataset = dataset_m[0]          
    x = dataset.iloc[:,0:1].values
    y = dataset.iloc[:,1:11].values
    
    error_indexes =[]
    for i in range(len(x)):
        if (x[i][0]=='timestamp'):
            error_indexes.append(i)
            
    x = np.delete(x,error_indexes,0)
    y = np.delete(y,error_indexes,0)
            
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = "NaN",strategy = "most_frequent",axis = 0)
    y = imputer.fit_transform(y)
    
    y_binary = np.int64(y>0)
    
    x_timestamp = np.float64(x)

#print(datetime.datetime.fromtimestamp(x_timestamp[0][0]))
#print(datetime.datetime.fromtimestamp(x_timestamp[1][0]))

    x_weekday = []
    x_hour = []
    x_minute = []
    x_second = []
    x_month = []
    for k in range(len(x_timestamp)):
        x_weekday.append((datetime.datetime.fromtimestamp(x_timestamp[k][0])).weekday())
        x_hour.append((datetime.datetime.fromtimestamp(x_timestamp[k][0])).hour)
        x_minute.append((datetime.datetime.fromtimestamp(x_timestamp[k][0])).minute)
        x_second.append((datetime.datetime.fromtimestamp(x_timestamp[k][0])).second)
        x_month.append((datetime.datetime.fromtimestamp(x_timestamp[k][0])).month)    
    x_features = np.c_[x_timestamp,x_month]
    x_features = np.c_[x_features,x_weekday]
    x_features = np.c_[x_features,x_hour]
    x_features = np.c_[x_features,x_minute]
    x_features = np.c_[x_features,x_second]

#x_features=[]
#y_binary = []

    
    
    length = len(dataset_m)
    for i in range(1,length):
        print(i," -->")
        temp_dataset = dataset_m[i]
        x_temp = temp_dataset.iloc[:,0:1].values
        y_temp = temp_dataset.iloc[:,1:11].values

        error_indexes_temp =[]
        for i in range(len(x_temp)):
            if (x_temp[i][0]=='timestamp'):
                error_indexes_temp.append(i)

        x_temp = np.delete(x_temp,error_indexes_temp,0)
        y_temp = np.delete(y_temp,error_indexes_temp,0)

        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values = "NaN",strategy = "most_frequent",axis = 0)
        y_temp = imputer.fit_transform(y_temp)
        y_binary_temp = np.int64(y_temp>0)
        x_timestamp_temp = np.float64(x_temp)
        
        temp_x_weekday = []
        temp_x_hour = []
        temp_x_minute = []
        temp_x_second = []
        temp_x_month = []
        
        for k in range(len(x_temp)):
            temp_x_weekday.append((datetime.datetime.fromtimestamp(x_timestamp_temp[k][0])).weekday())
            temp_x_hour.append((datetime.datetime.fromtimestamp(x_timestamp_temp[k][0])).hour)
            temp_x_minute.append((datetime.datetime.fromtimestamp(x_timestamp_temp[k][0])).minute)
            temp_x_second.append((datetime.datetime.fromtimestamp(x_timestamp_temp[k][0])).second)
            temp_x_month.append((datetime.datetime.fromtimestamp(x_timestamp_temp[k][0])).month)      
        temp_x_features = np.c_[x_timestamp_temp,temp_x_month]
        temp_x_features = np.c_[temp_x_features,temp_x_weekday]
        temp_x_features = np.c_[temp_x_features,temp_x_hour]
        temp_x_features = np.c_[temp_x_features,temp_x_minute]
        temp_x_features = np.c_[temp_x_features,temp_x_second]
        
        x_features = np.r_[x_features,temp_x_features]
        y_binary = np.r_[y_binary,y_binary_temp]
        y = np.r_[y,y_temp]

    return x_features,y_binary,y

# now retreiving all the addresses to import the files
def address_finder():
    list_of_building_names = os.listdir()
    address_of_greend_dataset = os.getcwd()
    address_of_all_building_folders_inside = []
    ## this loop extracts the address of all buildings 
    for i in range(len(list_of_building_names)):
        address_of_all_building_folders_inside.append(address_of_greend_dataset+'/'+list_of_building_names[i])
    # this loop extracts the addresses of all the folders inside the building folder    
    list_of_address_inside_building = []
    for i in range(len(address_of_all_building_folders_inside)):
        temp_address = os.listdir() 
        len_address = len(temp_address)
        temp_list = os.listdir(address_of_all_building_folders_inside[i])
        list_of_address_inside_building.append([])
        for k in range(len(temp_list)):
            list_of_address_inside_building[i].append(address_of_all_building_folders_inside[i]+'/'+temp_list[k])
    csv = '.csv'     
    csv_address_list = []
    #this loop extracts all the address of all the csv file
    for i in range(len(list_of_address_inside_building)):
        no_of_month = len(list_of_address_inside_building[i])
        csv_address_list.append([])
        for k in range(no_of_month):
            no_of_days = len(os.listdir(list_of_address_inside_building[i][k]))
            csv_address_list[i].append([])
            temp_list1 = os.listdir(list_of_address_inside_building[i][k])
            for j in range(no_of_days):
                csv_address_list[i][k].append(list_of_address_inside_building[i][k]+'/'+temp_list1[j])
    return csv_address_list                    

csv_address_list = address_finder()
building_2_year_csv_address = csv_address_list[2]
no_month_data_of_building_2 = len(building_2_year_csv_address)
building_2_year_address = []
for i in range(no_month_data_of_building_2):
    building_2_year_address = building_2_year_csv_address[i]+building_2_year_address

dataset_year= []
for i in range(len(building_2_year_address)):
    dataset_year.append(pd.read_csv(building_2_year_address[i]))
   
x_year,y_year,y_year_values = data_preprocessing(dataset_year)

from sklearn.externals import joblib
joblib.dump(x_year,"x_year.pkl")
joblib.dump(y_year,"y_year.pkl")
joblib.dump(y_year_values,"y_year_values.pkl")
joblib.dump(dataset_year,"dataset_year.pkl")
