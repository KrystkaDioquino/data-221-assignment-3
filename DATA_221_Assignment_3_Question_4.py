import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder

#This loads the kidney_disease.csv file
kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")

#Cleans the data frame. This handle missing values. This also fixes ValueError
kidney_disease_data_frame = kidney_disease_data_frame.dropna()

#Converts the columns with string values to integers to allow mathematical computations
encoder = OrdinalEncoder()
kidney_disease_text_columns = kidney_disease_data_frame.select_dtypes(include=["object", "string"]).columns
kidney_disease_data_frame[kidney_disease_text_columns] = encoder.fit_transform(
    kidney_disease_data_frame[kidney_disease_text_columns])

#This is a feature matrix that does not include the classification column
X = kidney_disease_data_frame.drop(columns =["classification"])

#Creates a label vector using the classification column
y = kidney_disease_data_frame["classification"]

#Splits the dataset into 70% training set and 30% testing set with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.30, random_state = 0)

#Sets the number of neighbors
number_of_neighbors = 5
knn_model = KNeighborsClassifier(n_neighbors = number_of_neighbors)

#Trains the model with the training data
trained_knn_model = knn_model.fit(X_train, y_train)

#Uses the trained model to make predictions on the testing data
y_predictions = trained_knn_model.predict(X_test)

#Calculates the metrics
confusion_matrix = confusion_matrix(y_test, y_predictions)
accuracy = accuracy_score(y_test, y_predictions)
precision = precision_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions)
f1_score = f1_score(y_test, y_predictions)

#Displays the results
print(f"Confusion Matrix: \n {confusion_matrix}\n"
      f"Accuracy: {accuracy:.2f}\n"
      f"Precision: {precision:.2f}\n"
      f"Recall: {recall:.2f}\n"
      f"F1-score: {f1_score:.2f}")

"""
In the context of kidney disease prediction, a True Positive happens when the model correctly predicts 
that a patient has CKD, while a True Negative happens when the model correctly predicts that a patient 
does not have CKD. A False Positive happens when the model predicts that a patient has CKD, but in reality, the patient is 
healthy, which can lead to unnecessary treatment. A False Negative happens when the model predicts that a 
patient does not have CKD, but the patient actually has the disease, which is dangerous as treatment may be delayed or not given. 

Accuracy only measures the proportion of correct predictions the model made out of all predictions and does not consider between 
the types of errors the model makes, such as False Negative and False Positive. While accuracy matters, recall is the 
the most important as this measures that no patient with kidney disease are missed.
"""