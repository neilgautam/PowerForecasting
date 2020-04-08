from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


#importing the pickle files in the variable 
x_year = joblib.load("x_year.pkl")
y_year = joblib.load("y_year.pkl")
y_year_values = joblib.load("y_year_values.pkl")
dataset_year = joblib.load("dataset_year.pkl")
x_year = x_year[:,1:]

x_dayname =['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
equipment = ['euipment 1','equipment 2','equipment 3','equipment 4','equipment 5','equipment 6','equipment 7','equipment 8','equipment 9']
for s in range(7):
    x_day_index = []
    for i in range(len(x_year)):
        if x_year[i][1]==s:
            x_day_index.append(i)
            
            
    x_day = x_year[x_day_index,:]
    y_day = y_year_values[x_day_index,:]        
    
    temp_index =[]
    for i in range(len(x_day)):
        if x_day[i][0]==12:
            temp_index.append(i)
        
    x_test_day = x_day[temp_index,:]
    y_test_day = y_day[temp_index,:]        
        

    temp_index_not =[]
    for i in range(len(x_day)):
        if x_day[i][0]!=12:
           temp_index_not.append(i)
        
    x_train_day  = x_day[temp_index_not,:]
    y_train_day =  y_day[temp_index_not,:]  
    
    
    for i in range(9):
        print("training neural network for {} for {}".format(x_dayname[s],equipment[i]))
        classifier = Sequential()
        classifier.add(Dense(output_dim = 128,activation = 'relu',input_dim = 5))
        classifier.add(Dense(output_dim = 128,activation = 'relu'))
        classifier.add(Dense(output_dim = 128,activation = 'relu'))
        classifier.add(Dense(output_dim = 1,activation = 'relu'))
        
        classifier.compile (optimizer = 'RMSprop',loss = 'mean_squared_error',metrics = ['accuracy'])
        classifier.fit(x_train_day,y_train_day[:,i],batch_size =1000,nb_epoch =150)
        y_pred = classifier.predict(x_test_day)
        
        joblib.dump(classifier,"classifier{0}{1}.pkl".format(x_dayname[s],equipment[i]))
        joblib.dump(y_pred,"prediction{0}{1}".format(x_dayname[s],equipment[i]))
        