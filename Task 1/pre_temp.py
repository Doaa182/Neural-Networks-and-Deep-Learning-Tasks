import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

def create_specific_dataset(selected_features,selected_classes ):
    Data = pd.read_excel('Dry_Bean_Dataset.xlsx')
    
     # to get each selected features name
    x1_name,x2_name=selected_features.split(' and ')

    # to get each selected cls name 
    class1_name,class2_name=selected_classes.split(' and ')
    
    # data set which contains rows which are in class  class1_name
    data_With_selected_class1=Data.loc[(Data.Class==class1_name)] 
    
    # data set which contains rows which are in class  class2_name
    data_With_selected_class2=Data.loc[(Data.Class==class2_name)]   
    
    # data set contains selected feature 1 column,selected feature 2 column,class=selected class1
    data_With_selected_class1=pd.DataFrame({x1_name:data_With_selected_class1.loc[:,x1_name],
                                            x2_name:data_With_selected_class1.loc[:,x2_name],
                                            'Class':data_With_selected_class1.iloc[:,5]})
    # data set contains selected feature 1 column,selected feature 2 column,class=selected class2
    data_With_selected_class2=pd.DataFrame({x1_name:data_With_selected_class2.loc[:,x1_name],
                                            x2_name:data_With_selected_class2.loc[:,x2_name],
                                            'Class':data_With_selected_class2.iloc[:,5]})
    
    # encoding class column in data_With_selected_class1
    data_With_selected_class1=preprocess_data(data_With_selected_class1,class1_name,class2_name)

    # encoding class column in data_With_selected_class2
    data_With_selected_class2=preprocess_data(data_With_selected_class2,class1_name,class2_name)
    
    if data_With_selected_class1 is None or data_With_selected_class2 is None:
        return None  # Exit if there's an issue with data conversion
    
    X_class1_dataset=data_With_selected_class1.iloc[:,0:2]
    Y_class1_dataset=data_With_selected_class1.iloc[:,2]
    
    X_class2_dataset=data_With_selected_class2.iloc[:,0:2]
    Y_class2_dataset=data_With_selected_class2.iloc[:,2]
    
    return  X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset


def preprocess_data(Data,class1_name,class2_name):
       
    # filling mising values
    if Data.isnull().values.any():
        Data.fillna(Data.mean(), inplace=True)
    
    warnings.filterwarnings('ignore') 
    # # encoding class clmn
    global class_map
    class_map = {class1_name:-1, class2_name:1}
    Data['Class'] = Data['Class'].map(class_map)
    
    min_x1=min(Data.iloc[:,0])
    max_x1=max(Data.iloc[:,0])
    
    min_x2=min(Data.iloc[:,1])
    max_x2=max(Data.iloc[:,1])
    
    Data.iloc[:,0]=(Data.iloc[:,0]-min_x1) /(max_x1-min_x1)
    Data.iloc[:,1]=(Data.iloc[:,1]-min_x2) /(max_x2-min_x2)
    
    
    # # # scaling features to be similar scaling
    # features = Data.drop('Class', axis=1)
    # the_mean = features.mean()
    # the_std = features.std()
    # scaling_features = (features - the_mean) / the_std
    # Data[features.columns] = scaling_features

    return Data


def split_data(X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset):
    
    x_train_class1,x_test_class1,y_train_class1,y_test_class1=train_test_split(X_class1_dataset,Y_class1_dataset
                                                  ,test_size=0.40,shuffle=False) 

    x_train_class2,x_test_class2,y_train_class2,y_test_class2=train_test_split(X_class2_dataset,Y_class2_dataset
                                                  ,test_size=0.40,shuffle=False) 

    Xtrain=pd.concat([x_train_class1,x_train_class2])
    Ytrain= pd.concat([y_train_class1,y_train_class2])
    Xtest= pd.concat([x_test_class1,x_test_class2])
    Ytest= pd.concat([y_test_class1,y_test_class2])
    
    return Xtrain,Ytrain,Xtest,Ytest
    



