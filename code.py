import pandas as pd
import numpy as np

data=pd.read_csv('AI-DataTrain.csv')
data.head()

y=data.Q50
x=data.drop('Q50', axis=1)
import sklearn.model_selection as model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.80,test_size=0.20, random_state=1)
print ("X_train: ", x_train)
print ("y_train: ", y_train)
print("X_test: ", x_test)
print ("y_test: ", y_test)

n_question = x_train.shape[1]
print(n_question)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print('Accuracy of GNB classifier on training set:{:.2f}%'
      .format(gnb.score(x_train, y_train)*100))
print('Accuracy of GNB classifier on test set:{:.2f}%'
      .format(gnb.score(x_test, y_test)*100))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred= gnb.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(gnb, classes=['PASS','FAIL'])
visualizer.fit(x_train, y_train) # Fit the training data to the visualizer
visualizer.score(x_test, y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data


question_list= np.array([['Q1'], ['Q2'], ['Q3'], ['Q4'], ['Q5'], ['Q6'], ['Q7'], ['Q8'], ['Q9'], ['Q10']])





# Define input features :

#Q1-Q10 stdents:900-999
"""
input_features = np.array([[0,0,1,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,1,1,1,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,1,0,1,1,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1],
						   [1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,0,0,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1],
                           [1,1,0,0,1,0,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
                           [0,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,1,0,1,0,0,0,1,1,1,0,1,0,0,1,0,1,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1],
						   [1,1,1,1,1,1,1,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,0,1,1,1,0,1,1,0,0],
						   [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                           [0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,0],
						   [0,0,1,1,1,1,1,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,1,1,0,1,0,0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,0,0,1,1],
						   [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,0,1,0,0,0,1,1,1],
						   [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1]])
"""
#on train data output will be in output.xlsx
input_features = np.array([[0,0,1,0,1,1,1,0,0,1],
                           [1,0,1,1,0,1,1,1,1,0],
                           [1,1,0,0,1,0,1,1,0,0],
                           [0,0,0,0,1,1,1,0,0,1],
                           [1,1,1,1,1,1,1,0,1,0],
                           [0,0,0,0,0,1,0,0,0,1],
                           [0,1,0,0,0,0,0,0,1,1],
                           [0,0,1,1,1,1,1,0,0,0],
                           [1,1,1,1,1,1,1,1,1,0],
                           [1,1,1,1,1,1,1,1,1,1]])

print(input_features.shape)
print(input_features)
# Define target output :
target_output = np.array([[0,1,1,1,0,0,1,0,1,0]])
# Reshaping our target output into vector :
target_output = target_output.reshape(10, 1)
print(target_output.shape)
print(target_output)


#on test data output will be in result.xlsx
input_features_test = np.array([[0,1,0,1,0,0,0,0,1,0],
                           [0,0,0,1,1,0,0,1,0,0],
                           [1,1,1,0,0,0,1,1,1,1],
                           [0,0,0,0,1,0,0,0,0,0],
                           [1,1,1,1,0,1,0,1,0,0],
                           [0,0,0,0,0,0,0,0,1,1],
                           [0,1,0,0,1,1,0,1,1,1],
                           [1,1,0,0,0,1,0,1,0,0],
                           [1,1,1,1,1,0,1,1,0,1],
                           [1,1,1,0,1,1,1,1,1,1]])

print(input_features_test.shape)
print(input_features_test)
# Define target output :
target_output_test = np.array([[0,0,0,1,1,0,0,1,0,0]])  # Reshaping our target output into vector :
target_output_test = target_output_test.reshape(10, 1)
print(target_output_test.shape)
print(target_output_test)




# Define weights :
weights = np.array([[0.2], [0.2],[0.2], [0.2], [0.2], [0.2], [0.2], [0.2], [0.2],[0.2]])
print(weights.shape)
print(weights)
# Bias weight :
bias = 0.3
# Learning Rate :
lr = 0.05
# Sigmoid function :


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # Derivative of sigmoid function :


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))
    # Main logic for neural network :
    # Running our code 10000 times :
for epoch in range(10000):
    inputs = input_features
    # Feedforward input :
    pred_in = np.dot(inputs, weights) + bias
    # Feedforward output :
    pred_out = sigmoid(pred_in)
    # Backpropogation
    # Calculating error
    error = pred_out - target_output

    # Going with the formula :
    x = error.sum()
    print(x)

    # Calculating derivative :
    dcost_dpred = error
    dpred_dz = sigmoid_der(pred_out)

    # Multiplying individual derivatives :
    z_delta = dcost_dpred * dpred_dz
    # Multiplying with the 3rd individual derivative :
    inputs = input_features.T
    weights -= lr * np.dot(inputs, z_delta)
    # Updating the bias weight value :
    for i in z_delta:
        bias -= lr * i
        # Printing final weights:


print('weights: ',weights)
print("\n\n")
print('bias: ',bias)
# Taking inputs :  student 911 Q1-Q10
single_point = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 1])
## 1st step :
result1 = np.dot(single_point, weights) + bias
# 2nd step :
result2 = sigmoid(result1)
# Print final result
print('result of 1:',result2)

# Taking inputs :   student 912 Q1-Q10
#single_point = np.array([0,1,1,0,0,1,1,0,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1,1,0,1,0,0,0,1,1,1,0,1,1,0,1,0,1,0,0])  # 1st step :
single_point = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1])
result1 = np.dot(single_point, weights) + bias
# 2nd step :
result2 = sigmoid(result1)
# Print final result
print('result of 2:',result2)

# Taking inputs :   student 913 Q1-Q10
single_point = np.array([1, 0, 0, 0, 1, 1, 0, 1, 0, 1])
# 1st step :
result1 = np.dot(single_point, weights) + bias
# 2nd step :
result2 = sigmoid(result1)
# Print final result
print('result of 3:',result2)

"""
import csv
import numpy as np
weights = np.array(weights)

with open('result.xlsx', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in range(0, weights.shape[0]):
        myList = []
        myList.append(weights[row])
        writer.writerow(myList)

"""

# import xlsxwriter module
import xlsxwriter

workbook = xlsxwriter.Workbook('output.xlsx')

# By default worksheet names in the spreadsheet will be
# Sheet1, Sheet2 etc., but we can also specify a name.
worksheet = workbook.add_worksheet("My sheet")

# Some data we want to write to the worksheet.
scores = (['Question No.', 'Weights'],
	['Q1', weights[0]],
	['Q2', weights[1]],
	['Q3', weights[2]],
	['Q4', weights[3]],
    ['Q5', weights[4]],
    ['Q6', weights[5]],
    ['Q7', weights[6]],
    ['Q8', weights[7]],
    ['Q9', weights[8]],
    ['Q10', weights[9]]
)

# Start from the first cell. Rows and
# columns are zero indexed.
row = 0
col = 0

# Iterate over the data and write it out row by row.
for Question_no, weight in (scores):
    worksheet.write(row, col, Question_no)
    worksheet.write(row, col + 1, weight)
    row += 1

workbook.close()
