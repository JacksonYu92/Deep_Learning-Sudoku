## Final Project: Applied Deep Learning
### Project name: Sudoku
### Author: Qichun Yu
### Notebook: https://github.com/JacksonYu92/Deep_Learning-Sudoku/blob/main/Sudoku_1M-Final_Project.ipynb

## Table of Contents
1. [Introduction](#Abstract)  
    1.1. [Abstract](#Abstract)  
    1.2. [Use Case](#Use-Case)  
    1.3. [Load and Read Data](#Load-and-Read-Data)  
    1.4. [Data Cleaning](#Data-Cleaning)
2. [Preprocessing](#Preprocessing)  
    2.1. [Normalization](#Normalization)  
    2.2. [Data Splitting](#Data-Splitting)  
3. [Convolutional Neural Networks (CNNs)](#Convolutional-Neural-Networks-(CNNs))  
    3.1. [CNN Model 1](#CNN-Model-1)  
    3.2. [CNN Model 2](#CNN-Model-2)   
    3.3. [CNN Model 3](#CNN-Model-3)  
    3.4. [CNN Model 4](#CNN-Model-4)  
    3.5. [CNN Model 5](#CNN-Model-5)  
    3.6. [CNN Model 6](#CNN-Model-6)  
    3.7. [CNN Model 7](#CNN-Model-7)  
    3.8. [CNN Model 8](#CNN-Model-8)  
    3.9. [CNN Model 9](#CNN-Model-9)  
    3.10. [CNN Model 10](#CNN-Model-10)  
    3.11. [CNN Model 11](#CNN-Model-11)  
4. [Recurrent Neural Networks (RNN)](#Recurrent-Neural-Networks-(RNN))  
    4.1. [RNN Model 1](#RNN-Model-1)    
    4.2. [RNN Model 2](#RNN-Model-2)  
5. [Long Short-Term Memory (LSTM)](#Long-Short-Term-Memory-(LSTM))   
6. [Discussion](#Discussion)
7. [Conclusion](#Conclusion)

### Abstract

Sudoku is a number puzzle game that requires you to fill in digits 1 to 9. The game requires digits 1 to 9 to appear exactly once in each row, column and each of the nine 3x3 subgrids. The project experiment with different neural networks such as CNN, RNN, and LSTM. The data have been divided by 9 and subtracted by 0.5 to achieve zero mean-centred data. The CNN model that includes 9 convolution layers with 512 kernels works best with 95% of training accuracy. The study found that an increase in the number of epochs, number of layers, and number of neurons per layer can help improve the accuracy of the neural network model. Moreover, the dropout layer and maxpooling can help prevent overfitting. Adding strides of 3 x 3 is useful but requires large computing power. The main objective of this project is to build a deep learning model for a mobile app company that can analyze the grid of Sudoku to be filled, solve the Sudoku problem, and fill the grid. The convolution neural networks (CNN) is good at extracting features from the dataset and can be used to solve a sudoku game successfully. 

### Use Case
A mobile app company is building a classical Sudoku game. The development team is working on building a deep learning model which can analyze the grid of sudoku puzzles to be filled, solve the sudoku puzzles problem, and then automatic fill the grid in the end. The dataset includes 1 million Sudoku quizs with solution. This deep learning model will be part of a larger code as the backend of the Sudoku game application. 

### Citation

Kyubyong Park.(September, 2022). 1 million Sudoku games. Retrieved from https://www.kaggle.com/datasets/bryanpark/sudoku.

### Environment
Operating system: Windows Server 2019 atacenter, 64-bit<br>
GPU: Tesla V100-PCIE-16GB

### Discussion

[Return to top](#Final-Project:-Applied-Deep-Learning)

The convolution neural networks (CNN) is good at extracting features. An increase in the number of epochs, number of layers, and number of neurons per layer can help improve the accuracy of the model. In this project, we start with a simple CNN model with 3 conv2D layers. The model is overfitting. So we have tried adding a dropout layer or using maxpooling to prevent overfitting. Adding strides of (3x3) could be good considering valid sudoku must follow each of the nine 3X3 sub-squares and should contain 1-9 digits without repetition. However, models that have strides require longer training hours with more epochs.  

After testing all of the models, the **cnn_model_11** has the best performance in solving sudoku games with a training accuracy of 0.9540 and validation accuracy of 0.9827. 

Let's test the model to see whether it can solve a real sudoku game or not.


```python
train_pred_11 = cnn_model11.predict(x_train[0].reshape((9, 9)).reshape(1, 9, 9, 1)).argmax(-1)+1
train_pred_11
```




    array([[[8, 6, 4, 3, 7, 1, 2, 5, 9],
            [3, 2, 5, 8, 4, 9, 7, 6, 1],
            [9, 7, 1, 2, 6, 5, 8, 4, 3],
            [4, 3, 6, 1, 9, 2, 5, 8, 7],
            [1, 9, 8, 6, 5, 7, 4, 3, 2],
            [2, 5, 7, 4, 8, 3, 9, 1, 6],
            [6, 8, 9, 7, 3, 4, 1, 2, 5],
            [7, 1, 3, 5, 2, 8, 6, 9, 4],
            [5, 4, 2, 9, 1, 6, 3, 7, 8]]], dtype=int64)




```python
train_real_11 = y_train[0]+1
train_real_11
```




    array([[8, 6, 4, 3, 7, 1, 2, 5, 9],
           [3, 2, 5, 8, 4, 9, 7, 6, 1],
           [9, 7, 1, 2, 6, 5, 8, 4, 3],
           [4, 3, 6, 1, 9, 2, 5, 8, 7],
           [1, 9, 8, 6, 5, 7, 4, 3, 2],
           [2, 5, 7, 4, 8, 3, 9, 1, 6],
           [6, 8, 9, 7, 3, 4, 1, 2, 5],
           [7, 1, 3, 5, 2, 8, 6, 9, 4],
           [5, 4, 2, 9, 1, 6, 3, 7, 8]])




```python
train_pred_11 - train_real_11
```




    array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=int64)



There are all zeros means the model is predicting all 81 numbers correctly! <br>
Let's test it with one of the quizzes from the testing set, which the model hasn't seem before. 


```python
test_pred_11 = cnn_model11.predict(x_test[1].reshape((9, 9)).reshape(1, 9, 9, 1)).argmax(-1)+1
test_pred_11
```




    array([[[3, 1, 8, 4, 9, 2, 7, 6, 5],
            [7, 9, 6, 5, 1, 8, 2, 4, 3],
            [2, 5, 4, 3, 7, 6, 1, 8, 9],
            [8, 2, 9, 6, 4, 1, 5, 3, 7],
            [6, 7, 3, 2, 8, 5, 4, 9, 1],
            [5, 4, 1, 7, 3, 9, 6, 2, 8],
            [4, 3, 2, 8, 5, 7, 9, 1, 6],
            [1, 6, 5, 9, 2, 3, 8, 7, 4],
            [9, 8, 7, 1, 6, 4, 3, 5, 2]]], dtype=int64)




```python
test_real_11 = y_test[1]+1
test_real_11
```




    array([[3, 1, 8, 4, 9, 2, 7, 6, 5],
           [7, 9, 6, 5, 1, 8, 2, 4, 3],
           [2, 5, 4, 3, 7, 6, 1, 8, 9],
           [8, 2, 9, 6, 4, 1, 5, 3, 7],
           [6, 7, 3, 2, 8, 5, 4, 9, 1],
           [5, 4, 1, 7, 3, 9, 6, 2, 8],
           [4, 3, 2, 8, 5, 7, 9, 1, 6],
           [1, 6, 5, 9, 2, 3, 8, 7, 4],
           [9, 8, 7, 1, 6, 4, 3, 5, 2]])




```python
test_pred_11 - test_real_11
```




    array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=int64)



There are all zeros means the model is predicting all 81 numbers correctly!

#### Confunsion matrix

Use the model to make predict all the testing data. 


```python
y_pred_11 = cnn_model11.predict(x_test)
```

Since we used normalization to transform the dataset before, we need to create functions to transform the prediction and the testing data back to the real board game with a dimension of (200000, 2). 


```python
def y_pred_func(len, model):
    list = []
    for i in range(len):
        test_pred = model.predict(x_test[i].reshape((9, 9)).reshape(1, 9, 9, 1)).argmax(-1)+1
        list.append((i, test_pred))
    return np.array(list)
            
```


```python
y_pred_11 = y_pred_func(x_test.shape[0], cnn_model11)
```

    C:\Users\DSMLAzure\AppData\Local\Temp\2\ipykernel_4852\1863722690.py:6: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return np.array(list)
    


```python
print(y_pred_11[0])
```

    [0 array([[[4, 9, 7, 3, 2, 5, 6, 8, 1],
               [3, 1, 2, 8, 7, 6, 5, 9, 4],
               [5, 6, 8, 9, 1, 4, 3, 7, 2],
               [9, 8, 4, 6, 3, 2, 1, 5, 7],
               [1, 5, 6, 7, 4, 8, 9, 2, 3],
               [7, 2, 3, 5, 9, 1, 4, 6, 8],
               [6, 3, 1, 2, 8, 9, 7, 4, 5],
               [8, 7, 5, 4, 6, 3, 2, 1, 9],
               [2, 4, 9, 1, 5, 7, 8, 3, 6]]], dtype=int64)]
    


```python
y_pred_11.shape
```




    (200000, 2)




```python
def y_real_func(data):
    len = data.shape[0]
    list = []
    for i in range(len):
        y_real = data[i] +1
        list.append((i, y_real))
    return np.array(list)
```


```python
y_real_11 = y_real_func(y_test)
```

    C:\Users\DSMLAzure\AppData\Local\Temp\2\ipykernel_4852\1141021980.py:7: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return np.array(list)
    


```python
y_real_11.shape
```




    (200000, 2)




```python
y_real_11[0][1]
```




    array([[4, 9, 7, 3, 2, 5, 6, 8, 1],
           [3, 1, 2, 8, 7, 6, 5, 9, 4],
           [5, 6, 8, 9, 1, 4, 3, 7, 2],
           [9, 8, 4, 6, 3, 2, 1, 5, 7],
           [1, 5, 6, 7, 4, 8, 9, 2, 3],
           [7, 2, 3, 5, 9, 1, 4, 6, 8],
           [6, 3, 1, 2, 8, 9, 7, 4, 5],
           [8, 7, 5, 4, 6, 3, 2, 1, 9],
           [2, 4, 9, 1, 5, 7, 8, 3, 6]])




```python
y_pred_11[0][1][0]
```




    array([[4, 9, 7, 3, 2, 5, 6, 8, 1],
           [3, 1, 2, 8, 7, 6, 5, 9, 4],
           [5, 6, 8, 9, 1, 4, 3, 7, 2],
           [9, 8, 4, 6, 3, 2, 1, 5, 7],
           [1, 5, 6, 7, 4, 8, 9, 2, 3],
           [7, 2, 3, 5, 9, 1, 4, 6, 8],
           [6, 3, 1, 2, 8, 9, 7, 4, 5],
           [8, 7, 5, 4, 6, 3, 2, 1, 9],
           [2, 4, 9, 1, 5, 7, 8, 3, 6]], dtype=int64)



Create a function to calculate the accuracy for each position. 


```python
def count_grid_acc(actual, predicted):
    result = np.zeros((9,9), dtype=np.int)
    for i in range(y_real_11.shape[0]):
        for j in range(9):
            for k in range(9):
                if actual[i][1][j][k] == predicted[i][1][0][j][k]:
                    result[j][k] += 1
    return result
 
```


```python
cga = count_grid_acc(y_real_11, y_pred_11)
```

    C:\Users\DSMLAzure\AppData\Local\Temp\2\ipykernel_4852\3278397974.py:2: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      result = np.zeros((9,9), dtype=np.int)
    


```python
ticks = [1,2,3,4,5,6,7,8,9]

plt.figure(figsize=(20,15))
sns.heatmap(cga, annot=True, fmt='g',xticklabels = ticks, yticklabels = ticks)
plt.title('Number of correct prediction for each position')
plt.xlabel('Predicted')
plt.ylabel('Actual')
```




    Text(220.72222222222223, 0.5, 'Actual')




    
![png](output_252_1.png)
    


It looks like for each position, the model makes the correct prediction of about 196,500 out of 200,000. It looks pretty high, let's build a function to check the accuracy for each position. 


```python
def count_grid_position_acc(cga):
    sum = y_test.shape[0]
    acc = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            acc[i][j] = round(cga[i][j] / sum, 4)
    return acc       
```


```python
cga_acc = count_grid_position_acc(cga)
```


```python
ticks = [1,2,3,4,5,6,7,8,9]

plt.figure(figsize=(20,15))
sns.heatmap(cga_acc, annot=True, fmt='g',xticklabels = ticks, yticklabels = ticks)
plt.title('Accuracy of correct prediction for each position')
plt.xlabel('Predicted')
plt.ylabel('Actual')
```




    Text(220.72222222222223, 0.5, 'Actual')




    
![png](output_256_1.png)
    


Looks like all positions are around 98% correct. <br>

Let's build a confusion matrix function to evaluate the model. The **confusion matrix** is a summary of prediction results for a classification problem. We want to evaluate to check if the model can classify the digit 1 to 9 correctly. 


```python
def confusion_matrix_func(actual, predicted):
    result = np.zeros((9,9), dtype=np.int)
    for i in range(y_real_11.shape[0]):
        for j in range(9):
            for k in range(9):
                if predicted[i][1][0][j][k] == actual[i][1][j][k]:
                    result[actual[i][1][j][k]-1][actual[i][1][j][k]-1] += 1
                else:
                    result[actual[i][1][j][k]-1][predicted[i][1][0][j][k]-1] += 1
    return result
        
```


```python
cm = confusion_matrix_func(y_real_11, y_pred_11)
```

    C:\Users\DSMLAzure\AppData\Local\Temp\2\ipykernel_4852\2344926541.py:2: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      result = np.zeros((9,9), dtype=np.int)
    


```python
cm
```




    array([[1767456,    4197,    3986,    4217,    4448,    4081,    3857,
               3983,    3775],
           [   3863, 1769048,    3762,    4067,    4135,    3932,    3871,
               3740,    3582],
           [   3741,    4121, 1768841,    4049,    4057,    3970,    3750,
               3817,    3654],
           [   3958,    4046,    3778, 1769192,    4181,    3814,    3814,
               3691,    3526],
           [   4029,    4329,    4013,    4305, 1767418,    4002,    4040,
               3995,    3869],
           [   4098,    4223,    3987,    4261,    4377, 1767346,    3985,
               4079,    3644],
           [   3870,    4097,    3868,    4080,    4371,    3898, 1768543,
               3767,    3506],
           [   3660,    3710,    3635,    3925,    3913,    3680,    3686,
            1770314,    3477],
           [   3555,    3777,    3502,    3788,    3854,    3675,    3525,
               3422, 1770902]])




```python
plt.figure(figsize=(20,15))
sns.heatmap(cm, annot=True, fmt='d',xticklabels = ticks, yticklabels = ticks)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
```




    Text(220.72222222222223, 0.5, 'Actual Label')




    
![png](output_261_1.png)
    


The y-axis is the actual label and the x-axis is the predicted label. For example, the first number 1,767,456 indicates that the model predicts the number is 1 when it is actually one. This is also called a true positive. However, the second of the first row indicates the model predicts 4197 times that the digit is 2 but it is actually 1. This is also called False Negative. The first number of the second row indicates the model predicts 3863 times that the digit is 1 but it is actually 2. 

Let's normalize the confusion matrix to have a better idea of how it performs.


```python
norm_cm = np.round(cm/cm.astype(np.float).sum(axis=1), 4)
```
    


```python
plt.figure(figsize=(20,15))
sns.heatmap(norm_cm, annot=True, fmt='g',xticklabels = ticks, yticklabels = ticks)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
```




    Text(220.72222222222223, 0.5, 'Actual Label')




    
![png](output_265_1.png)
    


We can see there are around 98% of times the model predicts the digit is 1 when it is actually 1. 


```python
df = pd.DataFrame(cm, index= ['1','2','3','4','5','6','7','8','9'], columns=['1','2','3','4','5','6','7','8','9'], dtype=int)
df.da.export_metrics()
```




<div>

    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>micro-average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy</th>
      <td>0.996091</td>
      <td>0.996083</td>
      <td>0.996192</td>
      <td>0.996080</td>
      <td>0.995931</td>
      <td>0.996068</td>
      <td>0.996174</td>
      <td>0.996285</td>
      <td>0.996412</td>
      <td>0.996146</td>
    </tr>
    <tr>
      <th>f1</th>
      <td>0.982403</td>
      <td>0.982382</td>
      <td>0.982861</td>
      <td>0.982370</td>
      <td>0.981693</td>
      <td>0.982296</td>
      <td>0.982778</td>
      <td>0.983287</td>
      <td>0.983852</td>
      <td>0.982658</td>
    </tr>
    <tr>
      <th>false_discovery_rate</th>
      <td>0.017113</td>
      <td>0.018040</td>
      <td>0.016968</td>
      <td>0.018143</td>
      <td>0.018512</td>
      <td>0.017266</td>
      <td>0.016969</td>
      <td>0.016934</td>
      <td>0.016130</td>
      <td>0.017342</td>
    </tr>
    <tr>
      <th>false_negative_rate</th>
      <td>0.018080</td>
      <td>0.017196</td>
      <td>0.017311</td>
      <td>0.017116</td>
      <td>0.018101</td>
      <td>0.018141</td>
      <td>0.017476</td>
      <td>0.016492</td>
      <td>0.016166</td>
      <td>0.017342</td>
    </tr>
    <tr>
      <th>false_positive_rate</th>
      <td>0.002137</td>
      <td>0.002257</td>
      <td>0.002120</td>
      <td>0.002270</td>
      <td>0.002315</td>
      <td>0.002156</td>
      <td>0.002120</td>
      <td>0.002118</td>
      <td>0.002016</td>
      <td>0.002168</td>
    </tr>
    <tr>
      <th>negative_predictive_value</th>
      <td>0.997740</td>
      <td>0.997850</td>
      <td>0.997836</td>
      <td>0.997860</td>
      <td>0.997737</td>
      <td>0.997733</td>
      <td>0.997816</td>
      <td>0.997938</td>
      <td>0.997979</td>
      <td>0.997832</td>
    </tr>
    <tr>
      <th>positive_predictive_value</th>
      <td>0.982887</td>
      <td>0.981960</td>
      <td>0.983032</td>
      <td>0.981857</td>
      <td>0.981488</td>
      <td>0.982734</td>
      <td>0.983031</td>
      <td>0.983066</td>
      <td>0.983870</td>
      <td>0.982658</td>
    </tr>
    <tr>
      <th>precision</th>
      <td>0.982887</td>
      <td>0.981960</td>
      <td>0.983032</td>
      <td>0.981857</td>
      <td>0.981488</td>
      <td>0.982734</td>
      <td>0.983031</td>
      <td>0.983066</td>
      <td>0.983870</td>
      <td>0.982658</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.981920</td>
      <td>0.982804</td>
      <td>0.982689</td>
      <td>0.982884</td>
      <td>0.981899</td>
      <td>0.981859</td>
      <td>0.982524</td>
      <td>0.983508</td>
      <td>0.983834</td>
      <td>0.982658</td>
    </tr>
    <tr>
      <th>sensitivity</th>
      <td>0.981920</td>
      <td>0.982804</td>
      <td>0.982689</td>
      <td>0.982884</td>
      <td>0.981899</td>
      <td>0.981859</td>
      <td>0.982524</td>
      <td>0.983508</td>
      <td>0.983834</td>
      <td>0.982658</td>
    </tr>
    <tr>
      <th>specificity</th>
      <td>0.997863</td>
      <td>0.997743</td>
      <td>0.997880</td>
      <td>0.997730</td>
      <td>0.997685</td>
      <td>0.997844</td>
      <td>0.997880</td>
      <td>0.997882</td>
      <td>0.997984</td>
      <td>0.997832</td>
    </tr>
    <tr>
      <th>true_negative_rate</th>
      <td>0.997863</td>
      <td>0.997743</td>
      <td>0.997880</td>
      <td>0.997730</td>
      <td>0.997685</td>
      <td>0.997844</td>
      <td>0.997880</td>
      <td>0.997882</td>
      <td>0.997984</td>
      <td>0.997832</td>
    </tr>
    <tr>
      <th>true_positive_rate</th>
      <td>0.981920</td>
      <td>0.982804</td>
      <td>0.982689</td>
      <td>0.982884</td>
      <td>0.981899</td>
      <td>0.981859</td>
      <td>0.982524</td>
      <td>0.983508</td>
      <td>0.983834</td>
      <td>0.982658</td>
    </tr>
  </tbody>
</table>
</div>



From the report above, we can see the average f1 score is 98% for this model. The F1 score encodes precision and recall. The score is high indicating the model is well performed. 

### Conclusion

In this study, we have experimented with different neural networks such as CNN, RNN, and LSTM. The above study has shown the possibility of solving sudoku games using Deep learning methods. The neural networks have better performance with zero-centred normalized data. The data have been divided by 9 and subtracted by 0.5 to achieve zero mean-centred data. The convolution neural networks (CNN) is good at extracting features. An increase in the number of epochs, number of layers, and number of neurons per layer can help improve the accuracy of the model. Dropout layer or maxpooling can help prevent overfitting. Adding strides of 3 x 3 could be useful but require large training hours and computing power. 

From the above experiments with different neural network models, the CNN model is able to solve a sudoku game but may still make some mistakes in the game. The cnn_model_11 is the optimal choice for model selection. The model includes 9 convolution layers with 512 kernels. A dropout layer of 0.1 was added to the model to prevent overfitting. After 5 epochs, the model generates a 95% of training accuracy. For each grid on the sudoku game board, the model is able to predict 98% correctly on unseen data. The model is also doing a great job to predict the digits 1 to 9 correctly. Overall, a CNN model with a certain number of convolution layers and a large number of neurons should be able to predict a classical sudoku game. 

[Return to top](#Final-Project:-Applied-Deep-Learning)


```python

```
