import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def read_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

def adjust_dataset_format(df):
    #transpose to make rows (individuals) and columns (genes)
    df=df.T
    #add "class" to the head of last column
    df.iloc[0,-1]="class"
    #take the first row as a header
    df.columns = df.iloc[0]
    #remove it from the df
    df = df[1:]
    return df

def divide_data_to_Xdata_and_Ytargetclass(df):
    X = df.drop(columns=["class"], axis = 1)
    Y = df["class"]
    return X,Y

def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def select_100_best_features(ls):
    features_index=[]
    index=0
    
    for i in range(0, 100): 
        max1 = 0         
        for j in range(len(ls)):     
            if ls[j] > max1: 
                max1 = ls[j];
                index=j                  
        ls.remove(max1); 
        features_index.append(index)          
    return features_index


def create_newdf_with_selected_features(df,fs):
    f= fs.scores_.tolist()
    features_index_selected = select_100_best_features(f)
    newDf = pd.DataFrame()
    for i in range (0,len(features_index_selected)):
        newDf.insert(i,"Feature"+str(i),df.iloc[:,features_index_selected[i]])
    newDf.insert(len(features_index_selected),"class",df.iloc[:,-1])
    return newDf

# Input  

dataset = "/reactfrontend/public/assets/dataset/testing1.csv"
df = read_data(dataset)
df = adjust_dataset_format(df)

# Feature Selection

x,y = divide_data_to_Xdata_and_Ytargetclass (df)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)

newDf =  create_newdf_with_selected_features (df,fs)

x,y = divide_data_to_Xdata_and_Ytargetclass (newDf)

# Model Training

k = 10
NMDmodel = SVC(kernel='linear', random_state = 1)
kf = KFold(n_splits=k, random_state=None)
acc_score = []

for train_index , test_index in kf.split(x):
    X_train , X_test = x.iloc[train_index,:],x.iloc[test_index,:]
    Y_train , Y_test = y[train_index] , y[test_index]
    NMDmodel.fit(X_train,Y_train)
    pred_values = NMDmodel.predict(X_test)
    acc = accuracy_score(Y_test,pred_values)
    acc_score.append(acc)

# User Input

input_file_name = "/reactfrontend/public/assets/dataset/testing1.csv"
test = read_data(input_file_name)

predictioned_value= NMDmodel.predict(test.iloc[:,1:])

predictioned_value