from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib  # Updated import statement
import seaborn as sns
import matplotlib.pyplot as plt


# Load the Breast Cancer dataset from scikit-learn
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a pipeline with StandardScaler and GradientBoostingClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=0))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(pipeline, 'trained_model.joblib')

# Predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save evaluation metrics to a text file
with open('evaluation_metrics.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}\n')
    file.write('Classification Report:\n')
    file.write(classification_rep)
    file.write('\nConfusion Matrix:\n')
    file.write(str(conf_matrix))

# Display the evaluation metrics
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)
print('Confusion Matrix:')
print(conf_matrix)

# Visualize the confusion matrix and save the plot
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=breast_cancer.target_names, yticklabels=breast_cancer.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
