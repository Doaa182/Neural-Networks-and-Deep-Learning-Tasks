import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') 
def create_specific_dataset():
    Data = pd.read_excel('Dry_Bean_Dataset.xlsx')
    
    data_With_class1=Data.loc[(Data.Class=='BOMBAY')] 

    data_With_class2=Data.loc[(Data.Class=='CALI')] 

    data_with_class3=Data.loc[(Data.Class=='SIRA')] 
    
    # encoding class column in data
    Data_c1=preprocess_data(data_With_class1)

    Data_c2=preprocess_data(data_With_class2)
    Data_c3=preprocess_data(data_with_class3)

    
    
    X_class1_dataset=Data_c1.iloc[:,0:5]
    Y_class1_dataset=Data_c1.iloc[:,5]
    # print('y1 clss1 ',Y_class1_dataset)
    
    X_class2_dataset=Data_c2.iloc[:,0:5]
    Y_class2_dataset=Data_c2.iloc[:,5]
    
    X_class3_dataset=Data_c3.iloc[:,0:5]
    Y_class3_dataset=Data_c3.iloc[:,5]
    
    return  X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset,X_class3_dataset,Y_class3_dataset


def preprocess_data(Data):
       
    # filling mising vues
    Data.fillna(Data.mean() , inplace= True)
    
    warnings.filterwarnings('ignore') 
    # # encoding class clmn
    global class_map
    class_map = {'BOMBAY':[1, 0, 0], 'CALI':[0, 1, 0],'SIRA':[0, 0, 1]}
    Data['Class'] = Data['Class'].map(class_map)
    
    min_x1=min(Data.iloc[:,0])
    max_x1=max(Data.iloc[:,0])
    
    min_x2=min(Data.iloc[:,1])
    max_x2=max(Data.iloc[:,1])
    
    min_x3=min(Data.iloc[:,2])
    max_x3=max(Data.iloc[:,2])
    
    min_x4=min(Data.iloc[:,3])
    max_x4=max(Data.iloc[:,3])
    
    min_x5=min(Data.iloc[:,4])
    max_x5=max(Data.iloc[:,4])
    
    Data.iloc[:,0]=(Data.iloc[:,0]-min_x1) /(max_x1-min_x1)
    Data.iloc[:,1]=(Data.iloc[:,1]-min_x2) /(max_x2-min_x2)
    Data.iloc[:,2]=(Data.iloc[:,2]-min_x3) /(max_x1-min_x3)
    Data.iloc[:,3]=(Data.iloc[:,3]-min_x4) /(max_x2-min_x4)
    Data.iloc[:,4]=(Data.iloc[:,4]-min_x5) /(max_x1-min_x5)
    
    
    # # # scaling features to be similar scaling
    # features = Data.drop('Class', axis=1)
    # the_mean = features.mean()
    # the_std = features.std()
    # scaling_features = (features - the_mean) / the_std
    # Data[features.columns] = scaling_features

    return Data


def split_data(X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset,X_class3_dataset,Y_class3_dataset):
    
    x_train_class1,x_test_class1,y_train_class1,y_test_class1=train_test_split(X_class1_dataset,Y_class1_dataset
                                                  ,test_size=0.40,shuffle=False) 

    x_train_class2,x_test_class2,y_train_class2,y_test_class2=train_test_split(X_class2_dataset,Y_class2_dataset
                                                  ,test_size=0.40,shuffle=False) 
    
    x_train_class3,x_test_class3,y_train_class3,y_test_class3=train_test_split(X_class3_dataset,Y_class3_dataset
                                                  ,test_size=0.40,shuffle=False) 

    Xtrain12=pd.concat([x_train_class1,x_train_class2])
    Xtrain=pd.concat([Xtrain12,x_train_class3])
    
    Ytrain12= pd.concat([y_train_class1,y_train_class2])
    Ytrain= pd.concat([Ytrain12,y_train_class3])
    
    Xtest12= pd.concat([x_test_class1,x_test_class2])
    Xtest= pd.concat([Xtest12,x_test_class3])
    
    Ytest12= pd.concat([y_test_class1,y_test_class2])
    Ytest= pd.concat([Ytest12,y_test_class3])
    
    return Xtrain,Ytrain,Xtest,Ytest
    



