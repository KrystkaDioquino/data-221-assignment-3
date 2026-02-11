import pandas as pd
from sklearn.model_selection import train_test_split

#This loads the kidney_disease.csv file
kidneyDiseaseDataFrame = pd.read_csv("kidney_disease.csv")

#This is a feature matrix that does not include the classification column
X = kidneyDiseaseDataFrame.drop(columns =["classification"])

#Creates a label vector using the classification column
y = kidneyDiseaseDataFrame["classification"]

#Splits the dataset into 70% training set and 30% testing set with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.30, random_state = 0)

#Displays how the training dataset and testing dataset is split up
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

"""
We do not train and test a model on the same data because accuracy problems occur, as the model
wouldn't be able to really learn the patterns and concept of the data. By using a different or unseen testing set, 
the model is tested on its capability to apply what it learned from the training set. This is also the purpose
of the testing set as it proves how well the model behaves and works when used or encounters new data.
"""