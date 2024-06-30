# Iris-Flower-Classification


ProblemStatement:
The iris flower, scientifically known as Iris, is a distinctive genus of flowering plants. Within this genus, there are three primary species: Iris setosa, Iris versicolor, and Iris virginica. These species exhibit variations in their physical characteristics, particularly in the measurements of their sepal length, sepal width, petal length, and petal width.

Objective:
The objective of this project is to develop a machine learning model capable of learning from the measurements of iris flowers and accurately classifying them into their respective species. The model's primary goal is to automate the classification process based on the distinct characteristics of each iris species.

Project Details:
Key Features:
The essential characteristics used for classification include:
Sepal Length (SepalLengthCm)
Sepal Width (SepalWidthCm)
Petal Length (PetalLengthCm)
Petal Width (PetalWidthCm)

Iris Species:
The dataset consists of iris flowers from three species:
Iris setosa
Iris versicolor
Iris virginica

Machine Learning Model:
The project involves the creation and training of multiple machine learning models to accurately classify iris flowers based on their measurements. The models used include Logistic Regression, Decision Tree, and Random Forest.

Significance:
This project demonstrates the application of machine learning techniques to classify iris species based on physical characteristics, which can have broader applications in botany, horticulture, and environmental monitoring.

Project Summary:

Project Description:
The Iris Flower Classification project focuses on developing a machine learning model to classify iris flowers into their respective species based on specific measurements. The project involves extensive data exploration, preprocessing, feature engineering, model training, evaluation, and visualization.

Objective:
The primary goal of this project is to leverage machine learning techniques to build a classification model that can accurately identify the species of iris flowers based on their measurements. The model aims to automate the classification process, offering a practical solution for identifying iris species.

Key Project Details:

Dataset: The dataset contains information about the measurements of iris flowers and their respective species.
Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features to prepare the dataset for modeling.
Feature Engineering: No additional features were created as the dataset already contains relevant measurements.
Model Selection: Training and evaluating multiple machine learning models to select the best one based on performance metrics.
Results:
Accuracy was chosen as the primary evaluation metric for the Iris Flower Classification model. The final list of models and their accuracies are as follows:

Logistic Regression: Accuracy
Decision Tree: Accuracy
Random Forest: Accuracy
Conclusion:
In the Iris Flower Classification project, the Random Forest model has been selected as the final prediction model due to its highest accuracy. The project aimed to classify iris flowers into three distinct species: Iris setosa, Iris versicolor, and Iris virginica. After extensive data exploration, preprocessing, and model evaluation, the following conclusions can be drawn:

Data Exploration: Through a thorough examination of the dataset, insights were gained into the characteristics and distributions of features. Petal length and petal width were found to be significant factors in differentiating between species.
Data Preprocessing: Data preprocessing steps, including encoding categorical variables and scaling numerical features, were performed to prepare the dataset for modeling.
Model Selection: After experimenting with various machine learning models, the Random Forest was chosen as the final model due to its simplicity, interpretability, and good performance in classifying iris species.
Model Training and Evaluation: The Random Forest model was trained on the training dataset and evaluated using accuracy metrics. The model demonstrated satisfactory accuracy in classifying iris species.
Practical Application:
The Iris Flower Classification model can be applied in real-world scenarios, such as botany and horticulture, to automate the identification of iris species based on physical characteristics.

How to Use:
Clone the repository:
git clone https://github.com/u-faizan/iris-flower-classification.git
cd iris-flower-classification
Place the Iris dataset (iris.csv) in the project directory.

Run the script:
python iris_flower_classification.py
Dependencies:
The following Python packages are required to run the code:

pandas
seaborn
matplotlib
scikit-learn
You can install the necessary packages using the following command:
pip install pandas seaborn matplotlib scikit-learn
Author:
Umar Faizan

License:
This project is licensed under the MIT License.
