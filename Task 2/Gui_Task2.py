from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore') 
from pre_temp2 import *
from temp_algorithm2 import*
def model():
    X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset,X_class3_dataset,Y_class3_dataset=create_specific_dataset()

    Xtrain,Ytrain,Xtest,Ytest=split_data(X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset,X_class3_dataset,Y_class3_dataset)
    
   
    num_HiddenLayer=int(hidden_entry.get())
    num_Neurons=int(neuron_entry.get())
    lrate=float(rate_entry.get())
    num_ephocs=int(epochs_entry.get())
    want_bias=b_var.get()
    global ActivationType
    if vall.get()=="sigmoid": 
        ActivationType='sigmoid'
    else:
        ActivationType='tanh_sigmoid'
    w,b=Train_algorithm(Xtrain,Ytrain,num_HiddenLayer,num_Neurons,lrate,num_ephocs,want_bias,ActivationType)

    Test_algorithm(Xtest,Ytest,num_HiddenLayer,num_Neurons,lrate,want_bias,ActivationType,w, b)
def Run_GuI():
    Form = tk.Tk()
    Form.geometry("400x400")
    Form.title("Neural Network Gui")
    
    
    # Hidden Layers
    hidden_label = tk.Label(Form, text="Enter Number of Hidden Layers:")
    hidden_label.pack()
    global hidden_entry
    hidden_entry = tk.Entry(Form)
    hidden_entry.pack(pady=4)
    
    # Neurons per Layer
    neuron_label = tk.Label(Form, text="Enter Number of Neurons per Layer:")
    neuron_label.pack()
    global neuron_entry
    neuron_entry = tk.Entry(Form)
    neuron_entry.pack(pady=4)
    
    # LEARNING RATE
    rate_label = tk.Label(Form, text="Enter learning rate:")
    rate_label.pack()
    global rate_entry
    rate_entry = tk.Entry(Form)
    rate_entry.pack(pady=4)
    
    # NUMBER OF EPOCHS
    epochs_label = tk.Label(Form, text="Enter number of epochs:")
    epochs_label.pack()
    global epochs_entry
    epochs_entry = tk.Entry(Form)
    epochs_entry.pack(pady=4)
    
    # BIAS 
    global b_var
    b_var = tk.BooleanVar()
    
    bias_checkbox = tk.Checkbutton(Form, text="Add Bias", variable = b_var)
    bias_checkbox.pack(pady=4)
    
    # CHOOSE ALGORITHM
    algorithm_label = tk.Label(Form, text="Choose the activation function:")
    algorithm_label.pack(pady=4)
    algo_var = tk.StringVar(value="sigmoid")
    global vall
    vall=tk.IntVar()
    perceptron_radio = tk.Radiobutton(Form, text="sigmoid", variable=algo_var, value="sigmoid")
    perceptron_radio.pack(pady=4)
    adaline_radio = tk.Radiobutton(Form, text="tanh_sigmoid", variable=algo_var, value="tanh_sigmoid")
    adaline_radio.pack(pady=4)
    
    
    train_button = tk.Button(Form, text="Train Model" , command = model)
    train_button.pack(pady=4)
    
    Form.mainloop()
Run_GuI()
