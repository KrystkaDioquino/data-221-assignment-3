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

#Creates a list where the accuracy scores are stored.
accuracy_list = []

#List of values of k used
values_of_k = [1,3,5,7,9]

#Goes through all the values of k and gets the accuracy score to be stored in a list
for num_neighbors in values_of_k:
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn_model.fit(X_train, y_train)

    y_predictions = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predictions)
    accuracy_list.append(accuracy)

#Creates an accuracy table where each value of k has an equivalent accuracy score
accuracy_table = pd.DataFrame({
    "k Value": values_of_k,
    "Accuracy Score": accuracy_list
})

#Displays teh accuracy table
print(accuracy_table)

#Initialize the values to be changed when the highest accuracy score has been found
highest_accuracy = 0
best_k = 0

#Loops through the whole accuracy list to find the highest accuracy to identify the best value of k
#that produced the highest accuracy score
for i in range(len(accuracy_list)):
    # Check if this accuracy is the highest so far
    if accuracy_list[i] > highest_accuracy:
        highest_accuracy = accuracy_list[i]
        best_k = values_of_k[i]

#Displays th result
print(f"\nThe k value of {best_k} produces the highest accuracy of {highest_accuracy:.2f}")


"""
Changing the value of k controls the number of nearest neighbors considered. When k is small, 
predictions are based on few nearby data points. This cause overfitting as it cannot generalize a
prediction using few information. On the other hand, when k is large, it uses many nearby data points 
to make a prediction causing underfitting as it fails to identify pattern from a broad scope.


"""