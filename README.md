# Machine-Learning-Pipeline-Method
This repository demonstrates the implementation of a machine learning classification task using the Breast Cancer dataset from scikit-learn. The project showcases two versions of the implementation: one without using a pipeline and another with a scikit-learn pipeline.

# Implementation Details
# Without Pipeline (ML_without_Pipeline.py)
Data Loading: The Breast Cancer dataset is loaded using scikit-learn's load_breast_cancer function.
Data Splitting: The dataset is split into training and testing sets using train_test_split.
Model Training: A Gradient Boosting Classifier is trained on the training data.
Model Evaluation: The model is evaluated on the testing set, and metrics such as accuracy, classification report, and confusion matrix are displayed.
Visualization: A confusion matrix and residuals distribution are visualized using seaborn and matplotlib.
Cleanup: Temporary files and extracted directories are removed after completion.
# With Pipeline (ML_with_Pipeline.py)
Data Loading: Similar to the non-pipeline version.
Data Splitting: Similar to the non-pipeline version.
Pipeline Creation: A scikit-learn pipeline is created using Pipeline with a StandardScaler for feature scaling and a GradientBoostingClassifier for training.
Model Training: The entire pipeline is fitted on the training data.
Model Saving: The trained model is saved using joblib.
Model Evaluation: Predictions are made on the testing set, and metrics are calculated as before.
Results Saving: Evaluation metrics are saved to a text file, and the confusion matrix plot is saved as an image.
Cleanup: Similar cleanup steps as the non-pipeline version.
# Usage
Execute ML_without_Pipeline.py and ML_with_Pipeline.py to observe the differences in implementation.
Review the generated evaluation metrics, confusion matrices, and saved model files.
