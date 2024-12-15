# Climate Prediction

### Overview
This project is part of the Machine Learning course at Cairo University, focused on implementing and evaluating Decision Tree, k-Nearest Neighbors (kNN), and Naïve Bayes algorithms. The task involves predicting whether it will rain using a modified weather forecast dataset obtained from Kaggle. The dataset contains six features and 2,500 observations.

### Problem Statement
The objective is to:
1. Preprocess the data for machine learning.
2. Implement and evaluate machine learning models using scikit-learn.
3. Build the kNN algorithm from scratch and compare its performance to the pre-built version.
4. Interpret and evaluate the models with respect to their decision-making processes and performance metrics.

### Dataset
The dataset consists of the following features:
- **Temperature**
- **Humidity**
- **Wind Speed**
- **Cloud Cover**
- **Pressure**
- **Rain** (target variable: 1 if it rains, 0 otherwise)

### Tasks

#### Task 1: Preprocessing
1. **Missing Data Handling**
   - Identify missing data in the dataset.
   - Handle missing data using two techniques:
     - Dropping missing values.
     - Replacing missing values with the feature’s mean.

2. **Feature Scaling**
   - Check if the data is on the same scale.
   - Apply standard scaling if necessary.

3. **Data Splitting**
   - Split the dataset into training and testing sets (80/20 split).

#### Task 2: Model Implementation
1. Implement Decision Tree, k-Nearest Neighbors (kNN), and Naïve Bayes using scikit-learn.
2. Evaluate models using accuracy, precision, and recall metrics.
3. Implement the k-Nearest Neighbors (kNN) algorithm from scratch.
4. Compare the custom kNN implementation with scikit-learn’s kNN using the evaluation metrics.

#### Task 3: Interpretation and Evaluation
1. **Effect of Data Handling**
   - Evaluate the performance of the models under different missing data handling techniques.

2. **Decision Tree Explanation**
   - Visualize the decision tree.
   - Explain the criteria and logic used at each node for predictions.

3. **Performance Metrics Report**
   - Compare the custom kNN implementation with scikit-learn’s kNN using at least five different values of `k`.
   - Provide detailed reports on the accuracy, precision, and recall of all models.

### Dependencies
This project uses the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

### Code Highlights
1. **Preprocessing**
   - Handling missing data by both dropping and replacing with mean.
   - Standard scaling of numeric features.

2. **Model Implementations**
   - Custom kNN implementation using Euclidean distance.
   - Comparison of custom kNN with scikit-learn’s implementation.
   - Decision Tree and Naïve Bayes implementation using scikit-learn.

3. **Visualization**
   - Decision tree visualization for interpretation.

### How to Run the Code
1. Install required libraries: `pip install numpy pandas scikit-learn matplotlib`
2. Place the dataset file `weather_forecast_data.csv` in the same directory as the code.
3. Run the Python script to preprocess data, train models, and evaluate performance.

### Results
- The project evaluates the impact of different missing data handling techniques.
- Performance of models is compared using multiple metrics (accuracy, precision, recall).
- Insights are drawn from the custom kNN implementation and decision tree visualization.

### Future Work
- Extend the dataset with additional features for better predictions.
- Explore advanced hyperparameter tuning techniques for improved model performance.
- Implement additional algorithms to enhance comparative analysis.


