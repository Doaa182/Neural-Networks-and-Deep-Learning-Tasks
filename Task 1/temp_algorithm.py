import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def initialize_weights(add_bais):
   weights=np.random.uniform(low=-3,high=1,size=(2))
   print(weights.shape)
   if  add_bais==False:
       bais=0
       return weights,bais
   else:
       bais=random.random()
       return weights,bais


def apply_activation_function(value):

    if(value>=0):
       return 1
    return -1



def train_percptron_algorithm(X,Y,lrate,ephocs,want_bais): 
    
  weights,bais=initialize_weights(want_bais)
  p_v=[]
  # print("TRAINGING DATA", X)
  for e in range(ephocs):
    for sample in range(X.shape[0]):   #loop over each row in x dataset

        x1=X.iloc[sample,0]
        x2=X.iloc[sample,1]
        x=np.array([x1,x2])
        target=Y.iloc[sample]
        
        
        net_value=np.dot(x,weights)+bais
        predicted_y=apply_activation_function(net_value)

        p_v.append(predicted_y)
        if predicted_y != target :
            error=target-predicted_y
           
            
            weights+=(lrate*error*x)
     
            if want_bais==False:
                     
                         continue # bais = 0 and will not be updated in all iterations
                         
            else:
                         
                         bais+=(lrate*error*1)
 # algorithm should returned weights which caused conversion to draw ( decision boundryline ) to classify classes 
 #confusion_matrix('train',Y,p_v)
 #print("EPOCHS",e)
 
  return weights ,bais 




def test_algorithm(X,Y,weights,bais):
    
    predicted_y_values=[]
    
    for sample in range(X.shape[0]):  
        x=X.iloc[sample,:]
        target=Y.iloc[sample]
        
        net_value=np.dot(x,weights)+bais
        predicted_y=apply_activation_function(net_value)
        
        # put predicted_y values at list to be used  at confusion matrix function
        predicted_y_values.append(predicted_y)

    confusion_matrix('test',Y,predicted_y_values)


def confusion_matrix(dataname,actual_y,predicted_y):
    tp=0
    tn=0
    fp=0
    fn=0
      # -1->negative , 1->positive#    
    for i in range(len(actual_y)):
      if ( (predicted_y[i] == actual_y.iloc[i]) and actual_y.iloc[i]==1): 
                tp+=1
      elif ( (predicted_y[i] == actual_y.iloc[i]) and actual_y.iloc[i]==-1): 
                tn+=1
      elif( (predicted_y[i] != actual_y.iloc[i]) and actual_y.iloc[i]==1 ):
            
                  fn+=1
      elif( (predicted_y[i] != actual_y.iloc[i]) and actual_y.iloc[i]==-1 ):
                fp+=1   
    print('---Confuision Matrix for ',dataname,' ------------')            
    print('-----------Actual Values------------')
    print('----Positive-----Negative-----------')
    print('P  ',tp,'\t\t\t',fp)
    print('N  ',fn,'\t\t\t',tn)
    print('------------------------------------')
    
    global Accuracy,Percision,Recall,F1_score
    Accuracy=(tp+tn)/(tp+fp+tn+fn)
    print(' Accuracy : ',Accuracy)
    
    # if ((tp+fp)==0):  
    #     Percision=0
    #     print(' Percision : ',Percision)
    # else:    
    #     Percision=tp/(tp+fp)
    #     print(' Percision : ',Percision)
        
        
    # if ((tp+fn)==0):
    #     Recall=0
    #     print(' Recall : ',Recall)
    # else:    
    #     Recall=tp/(tp+fn)
    #     print(' Recall : ',Recall)
        
        
    # if(Recall==0 or Percision==0):
    #     F1_score=0
    #     print(' F1_score : ',0)
    # else:
    #     F1_score=2/((1/Recall)+(1/Percision))
    #     print(' F1_score : ',F1_score)
        
    print(' _____________________________________________________')      



def plot_decision_boundary(selected_features,Xtrain,Ytrain,Xtest,Ytest, weights, b):
    n1,n2=selected_features.split(' and ')
    
 
    x1= np.linspace(Xtrain.iloc[:,0].min(), Xtrain.iloc[:,0].max(), 10)
    # print("X AXIS VALUES",x1);
    x2 = (-weights[0] * x1 - b) / weights[1]
    # print("Y AXIS VALUES",x2);
    plt.figure(figsize=(5,5))
    plt.title('Training Dataset')
    plt.scatter(Xtrain.iloc[:,0],Xtrain.iloc[:,1],marker='o',c=Ytrain)
    plt.grid()
   
    plt.plot(x1,x2, label='Decision Boundary')
    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.show()
    
    
    plt.figure(figsize=(5,5))
    plt.title('Testing Dataset')
    plt.scatter(Xtest.iloc[:,0],Xtest.iloc[:,1],marker='o',c=Ytest)
    plt.grid()
    plt.plot(x1,x2, label='Decision Boundary')
    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.show()
   

def adaline(X,Y,lrate,ephocs,want_bais,threshold):
        weights,bais=initialize_weights(want_bais)
        p_v=[]
        for e in range(ephocs):
          for sample in range(X.shape[0]):   #loop over each row in x dataset

              x1=X.iloc[sample,0]
              x2=X.iloc[sample,1]
              x=np.array([x1,x2])
              
              target=Y.iloc[sample]
              
              net_value=np.dot(x,weights)+bais
              predicted_y=net_value

              p_v.append(predicted_y)
              if predicted_y != target :
                  error=target-predicted_y
                  weights+=(lrate*error*x)
                  if want_bais==False:
                               continue # bais = 0 and will not be updated in all iterations
                  else:
                               bais+=(lrate*error*1)          
            
          for sample in range(X.shape[0]): 
             x1=X.iloc[sample,0]
             x2=X.iloc[sample,1]
             x=np.array([x1,x2])
             
             target=Y.iloc[sample]
             yhat=np.dot(x,weights)+bais
             error=target-yhat
             ##calculate mse
             mse=(error ** 2).mean()
             
          if(mse<threshold):
            break
          else:
            continue
        return weights,bais
    
    




    
   