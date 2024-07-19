import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import *
from PIL import Image, ImageTk
from pre_temp import *
from temp_algorithm import *

def train_model():
        # Get user inputs
        selected_features = feature_select.get()
        selected_classes = classes_select.get()
        learning_rate = float(rate_entry.get())
        num_of_epochs = int(epochs_entry.get())
        mse_threshold = float(mse_threshold_entry.get())
        bias = b_var.get()
        selected_algorithm = algo_var.get()
        
        # functions implmentaion are in preprocess.py
        X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset=create_specific_dataset(selected_features,selected_classes)
        Xtrain,Ytrain,Xtest,Ytest=split_data(X_class1_dataset,Y_class1_dataset,X_class2_dataset,Y_class2_dataset)

            
        if selected_algorithm=='Perceptron':
            print('You select Perceptron')
            w,bais=train_percptron_algorithm(Xtrain,Ytrain,learning_rate,num_of_epochs,bias)
            test_algorithm(Xtest, Ytest, w, bais)
            plot_decision_boundary(selected_features,Xtrain, Ytrain,Xtest,Ytest,w,bais)
        else:
             print('You select Adaline')
             w,bais=adaline(Xtrain,Ytrain,learning_rate,num_of_epochs,bias,mse_threshold)
             test_algorithm(Xtest, Ytest, w, bais)
             plot_decision_boundary(selected_features,Xtrain, Ytrain,Xtest,Ytest, w, bais)

    


# CREATE FORM
Form = tk.Tk()
Form.geometry("400x400")
Form.title("Neural Network Gui")

# making background image
bg_image = Image.open("background.webp")
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(Form, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


# FEATURE SELECTION
f_label = tk.Label(Form, text="Select Two Features")
f_label.grid(row=0, column=0, padx=10, pady=10)
feature_select = ttk.Combobox(Form, values=[
    "Area and Perimeter",
    "Area and MajorAxisLength",
    "Area and MinorAxisLength",
    "Area and roundnes",
    "Perimeter and MajorAxisLength",
    "Perimeter and MinorAxisLength",
    "Perimeter and roundnes",
    "MajorAxisLength and MinorAxisLength",
    "MajorAxisLength and roundnes",
    "MinorAxisLength and roundnes"
])
feature_select.set("Area and Perimeter")
feature_select.grid(row=0, column=1, padx=10, pady=10)


# CLASSES SELECTION
c_label = tk.Label(Form, text="Select Two Classes")
c_label.grid(row=1, column=0, padx=10, pady=10) 
classes_select = ttk.Combobox(Form , values=[
    'BOMBAY and CALI',
    'BOMBAY and SIRA',
    'CALI and SIRA'
])
classes_select.set('BOMBAY and CALI')
classes_select.grid(row=1, column=1, padx=10, pady=10) 

# LEARNING RATE
rate_label = tk.Label(Form, text="Enter learning rate:")
rate_label.grid(row=2, column=0, padx=10, pady=10)
rate_entry = tk.Entry(Form)
rate_entry.grid(row=2, column=1, padx=10, pady=10)

# NUMBER OF EPOCHS
epochs_label = tk.Label(Form, text="Enter number of epochs:")
epochs_label.grid(row=3, column=0, padx=10, pady=10)
epochs_entry = tk.Entry(Form)
epochs_entry.grid(row=3, column=1, padx=10, pady=10)

# MSE THRESHOLD
mse_threshold_label = tk.Label(Form, text="Enter MSE threshold:")
mse_threshold_label.grid(row=4, column=0, padx=10, pady=10)
mse_threshold_entry = tk.Entry(Form)
mse_threshold_entry.grid(row=4, column=1, padx=10, pady=10)

# BIAS 
b_var = tk.BooleanVar()
bias_checkbox = tk.Checkbutton(Form, text="Add Bias", variable = b_var)
bias_checkbox.grid()

# CHOOSE ALGORITHM
algorithm_label = tk.Label(Form, text="Choose the algorithm:")
algorithm_label.grid( padx=10, pady=10)
algo_var = tk.StringVar(value="Perceptron")
perceptron_radio = tk.Radiobutton(Form, text="Perceptron", variable=algo_var, value="Perceptron")
perceptron_radio.grid(row=6, column=1, padx=5, pady=8)
adaline_radio = tk.Radiobutton(Form, text="Adaline", variable=algo_var, value="Adaline")
adaline_radio.grid(row=6, column=2, padx=5, pady=8)


train_button = tk.Button(Form, text="Train Model" , command = train_model)
train_button.grid(padx=20, pady=20)

Form.mainloop()





