import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data (replace 'student_data.csv' with your actual file path)
# Example:data = pd.read_csv('C:/Users/YourUserName/Desktop/student_data.csv')
data = pd.read_csv('student_data.csv')

# Separating feature (X) and target variable (y)
X = data.drop('enrolled', axis=1)
y = data['enrolled']

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a decision tree classifier
model = DecisionTreeClassifier()

# Training the model
model.fit(X_train, y_train)

# predictions on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)