import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Preprocess_Task2 import *
import warnings
warnings.filterwarnings('ignore') 

def sigmoid(net):
    return 1.0 / (1.0+np.exp(-net))

def derivative_of_sigmoid(net):
    return sigmoid(net)*(1-sigmoid(net))

def tanh(net):
    return np.tanh(net)

def derivative_of_tanh(net):
    return 1-(np.tanh(net)**2)


def Max_1_Other_0(Predicted_Output):
    # print('----Max_1_Other_0 FUNCTION---')
    # Predicted=np.array(Predicted_Output)
    # print('Z FROM OUTPUT LAYER IS ',Predicted)
    Predicted=list(Predicted_Output)
    max_number=max(Predicted)
    Predicted_Output = [1 if digit == max_number else 0 for digit in Predicted]
    # print('Z FROM OUTPUT LAYER AFTER TRANSFORMATION IS ',Predicted_Output)
    # print('------------------------')
    return Predicted_Output
    
def intialize_weights(input_layer,num_HiddenLayer,num_Neurons,want_bias,all_weights,all_bais):
    # Intialize weights for input layer
    all_weights.append(np.random.uniform(low=-3,high=3,size=(len(input_layer), num_Neurons)))
    
    if want_bias==True:
        all_bais.append(np.random.uniform(low=-3,high=3,size=(num_Neurons,)))
    else:
        all_bais.append(np.zeros(shape=(num_Neurons,)))
    
    #Intializse weights for all hidden layers excpt last hidden layer
    for hiddenlayer in range(num_HiddenLayer-1):
      all_weights.append(np.random.uniform(low=-3,high=3,size=(num_Neurons, num_Neurons)))
      if want_bias==True:
          all_bais.append(np.random.uniform(size=(num_Neurons,)))
      else:
          all_bais.append(np.zeros(shape=(num_Neurons,)))
          
    # Initialize weights and biases for the last hidden  layer
    all_weights.append(np.random.uniform(low=-3,high=3,size=(num_Neurons, 3)))
    if want_bias==True:
         all_bais.append(np.random.uniform(size=(3,)))
    else :
         all_bais.append(np.zeros(shape=(3,)))
    # print('INTIALZIED WEIGHTS')     
    # print(all_weights)
    # print('BAIS') 
    # print(all_bais)    
    return   all_weights,all_bais



def calculate_Net_Z(num_HiddenLayer,outputs_from_layer,all_weights,all_bais,net_values,ActivationType):
  # print('---- CALCULATE NET & Z FUNCTION -----')
  for layer_number in range(num_HiddenLayer+1):
            global z
            net = np.dot(outputs_from_layer[layer_number], all_weights[layer_number])+all_bais[layer_number]
            net_values.append(net)
            if ActivationType=='sigmoid':
                    z = sigmoid(net)
            elif  ActivationType=='tanh_sigmoid': 
                    z = tanh(net) 
            if layer_number==num_HiddenLayer: # it's last hidden layer convert Z to binary number ,max will replaced by 1 other 0
                    binary_z=Max_1_Other_0(z)
            else:
                outputs_from_layer.append(z)
            
            
  # print('outputs_from_layer') 
  # print(outputs_from_layer)    
  # print('--------------') 
  # print('net_values') 
  # print(net_values)  
  return outputs_from_layer,net_values,binary_z,z

def OutPutLayer_SignalError(target,binary_predicted_output,net_values,ActivationType):
    # print('----OutPutLayer_SignalError---')
    net_result_from_last_hidden_layer=net_values[-1]
    # print('net_result_from_last_hidden_layer')
    # print(net_result_from_last_hidden_layer)
    global derivative_of_net
    if ActivationType=='sigmoid':
             derivative_of_net=derivative_of_sigmoid(net_result_from_last_hidden_layer)
    elif  ActivationType=='tanh_sigmoid':
              derivative_of_net=derivative_of_tanh(net_result_from_last_hidden_layer) 
    # print('derivative_of_net')
    # print(derivative_of_net)
    error=np.subtract(target,binary_predicted_output)
    # print('error')
    # print(error)
    Error_Signal_OutputLayer=error*derivative_of_net
    # print('OutPutLayer_SignalError')
    # print(Error_Signal_OutputLayer)
    return Error_Signal_OutputLayer

def HiddenLayer_SignalError(num_HiddenLayer,net_values,Error_signals,all_weights,ActivationType):
    # print('--HiddenLayer_SignalError---')
    layer_nymber=num_HiddenLayer
    while(layer_nymber>0)  :  
        # print('Layer number ',layer_nymber)
          
        net_result_from_hidden_layer=net_values[layer_nymber-1]
        # print('net of last layer is ')
        # print(net_result_from_hidden_layer)
        global derivative_of_net
        if ActivationType=='sigmoid':
                 derivative_of_net=(derivative_of_sigmoid(net_result_from_hidden_layer))
        elif  ActivationType=='tanh_sigmoid':
                  derivative_of_net=( derivative_of_tanh(net_result_from_hidden_layer) )
        # print('derivative_of_net')
        # print(derivative_of_net)          
                  
        # print('Error_signals[num_HiddenLayer-layer_nymber]')
        # print(Error_signals[num_HiddenLayer-layer_nymber])
        # print('all_weights[layer_nymber].T')
        # print(all_weights[layer_nymber].T)
        s=np.dot(Error_signals[num_HiddenLayer-layer_nymber],all_weights[layer_nymber].T)
        Error_signals.append(s*derivative_of_net)
        # print('Error signal in hidden layer ',layer_nymber)
        # print(s*derivative_of_net)
        layer_nymber-=1
    return Error_signals 

def UPDATE_WEIGHTS (all_weights,all_bais,outputs_from_layer,num_HiddenLayer,lrate,Error_signals,Z_output):
                # hidden layer 
                # print('---------UPDATE_WEIGHTS---------------')
                for layer_number in range(num_HiddenLayer,0,-1):
                    # print('layer_number ',layer_number)
                    # print('Error_signals[layer_number] ',Error_signals[num_HiddenLayer-layer_number])
                   
                    if layer_number==num_HiddenLayer:# UPDATE LAST HIDEEN LAYER WEIGHTS
                        # print('outputs_from_layer[layer_number-1] ',Z_output)
                        # print('OLD WEIGHT',all_weights[layer_number])
                        all_weights[layer_number]+=lrate*np.dot(Error_signals[num_HiddenLayer-layer_number],Z_output)
                        # print('NEW WEIGHT',all_weights[layer_number])
                        all_bais[layer_number]+=lrate*Error_signals[num_HiddenLayer-layer_number]
                    else :
                        # print('outputs_from_layer[layer_number-1] ',outputs_from_layer[layer_number])
                        # print('OLD WEIGHT',all_weights[layer_number])
                        all_weights[layer_number]+=lrate*np.dot(Error_signals[num_HiddenLayer-layer_number],outputs_from_layer[layer_number])
                        # print('NEW WEIGHT',all_weights[layer_number])
                        all_bais[layer_number]+=lrate*Error_signals[num_HiddenLayer-layer_number]
                return all_weights, all_bais

def Train_algorithm(X,Y,num_HiddenLayer,num_Neurons,lrate,num_ephocs,want_bias,ActivationType):
    global all_weights
    global all_bais
    global net_values
    global outputs_from_layer
    for e in range(num_ephocs):
        all_weights = []
        all_bais=[]
        predicted_output_for_confuison_matrix=[]
        true_output=[]
        for sample in range(X.shape[0]):
            net_values=[]
            outputs_from_layer=[]
            #~~~~~FORWARD STEP~~~~~
            #[1] Take inputs ang coresponding target from data set
            input_layer=list(X.iloc[sample,:])
            outputs_from_layer.append(input_layer)
            target=list(Y.iloc[sample])
            true_output.append(target)
            #[2] Initialize weights and biases for the layers
            all_weights,all_bais=intialize_weights(input_layer,num_HiddenLayer,num_Neurons,want_bias,all_weights,all_bais)
            
            #[3] Calculate Net & Z and (get binary_predicted_output vectore from outputlayer)
            # we will save net of each neuron in net_values list to use it in backward step
            outputs_from_layer,net_values,binary_predicted_output,Z_output=calculate_Net_Z(num_HiddenLayer,outputs_from_layer,all_weights,all_bais,net_values,ActivationType)
             
            predicted_output_for_confuison_matrix.append(binary_predicted_output)
            #~~~~~BACKWARD STEP~~~~~
            
            Error_Signal_OutputLayer=OutPutLayer_SignalError(target,binary_predicted_output,net_values,ActivationType)
            
            Error_signals=[]
            Error_signals.append(Error_Signal_OutputLayer)
            Error_signals=HiddenLayer_SignalError(num_HiddenLayer,net_values,Error_signals,all_weights,ActivationType)
            
            #UPDATE WEIGHTS 

            all_weights, all_bais=UPDATE_WEIGHTS(all_weights,all_bais,outputs_from_layer,num_HiddenLayer,lrate,Error_signals,Z_output)
    confusion_matrix('train ',true_output,predicted_output_for_confuison_matrix)        
    return all_weights, all_bais 
            

def Test_algorithm(X,Y,num_HiddenLayer,num_Neurons,lrate,want_bias,ActivationType,all_weights, all_bais):
     predicted_output=[]
     actual=[]
     for sample in range(X.shape[0]):
         #~~~~~FORWARD STEP~~~~~
         #[1] Take inputs ang coresponding target from data set
         outputs_from_layer=[]
         net_values=[]
         input_layer=list(X.iloc[sample,:])
         outputs_from_layer.append(input_layer)
         target=list(Y.iloc[sample])
         actual.append(target)
         

         
         #[3] Calculate Net & Z and (get binary_predicted_output vectore from outputlayer)
         # we will save net of each neuron in net_values list to use it in backward step
         outputs_from_layer,net_values,binary_predicted_output,z=calculate_Net_Z(num_HiddenLayer,outputs_from_layer,all_weights,all_bais,net_values,ActivationType)
         predicted_output.append(binary_predicted_output)
     confusion_matrix('test ',actual,predicted_output)   
         

from sklearn.metrics import confusion_matrix
def confusion_matrix(title,actual_y,predicted_y):
    print('ACCURACY for ',title)
    tp=0
    fn=0
    for i in range(len(actual_y)):
        actual=actual_y[i]
        if (predicted_y[i] == list(actual) ) :
            tp+=1
        else:
            fn+=1
    print('ACCURAY = ',tp/(tp+fn))       
    print(' _____________________________________________________')      

