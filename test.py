import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# load the Dataset
df=pd.read_csv('fraud_data.csv')    
# Preprocess the data
x=df.drop('is_fraud', axis=1)   
y=df['is_fraud']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=44)   
# Now we will make predictions on the testing set
y_pred = log_reg.predict(x_test)
# Assess the performance of the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification:')
print(classification_report(y_test, y_pred))
print('Confusion:')
print(confusion_matrix(y_test, y_pred))