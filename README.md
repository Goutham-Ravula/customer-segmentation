# Customer Classification Project

![License](https://img.shields.io/badge/license-MIT-blue)

## Introduction

Customer segmentation is a crucial strategy in business, aiming to categorize consumers based on shared characteristics. This practice aids companies in tailoring their marketing approaches, product offerings, and customer interactions, thereby enhancing overall efficiency and customer satisfaction.

This project delves into the development of a robust customer segmentation system, leveraging machine learning techniques. The primary objective is to create a classifier capable of accurately categorizing consumers into predefined clusters during their initial interactions. This proactive segmentation allows businesses to understand and address customer needs more effectively, paving the way for personalized services and targeted marketing efforts.

The project unfolds through a systematic evaluation of various machine learning classifiers implemented using scikit-learn. Each classifier undergoes a comprehensive tuning process to identify the one that demonstrates optimal performance in assigning clients to their respective categories

## Dataset

This dataset comprises eight variables, each representing distinct aspects of the transactions:

1. **InvoiceNo**: An integral, six-digit nominal code uniquely assigned to each transaction.
2. **StockCode**: A five-digit integral nominal code uniquely assigned to each distinct product.
3. **Description**: Nominal field indicating the name of the product.
4. **Quantity**: Numeric field denoting the quantity of each product per transaction.
5. **InvoiceDate**: Numeric field representing the date and time of each transaction generation.
6. **UnitPrice**: Numeric field indicating the unit price of the product in sterling.
7. **CustomerID**: Nominal, five-digit integral code uniquely assigned to each customer.
8. **Country**: Nominal field specifying the name of the country where each customer resides

## Project Objectives

1. **Customer Segmentation:**
   - *Objective:* Implement customer segmentation techniques to categorize consumers into distinct clusters based on their transactional behavior.
   - *Rationale:* Understanding customer segments enables businesses to tailor marketing strategies, personalize customer experiences, and optimize product offerings.

2. **Machine Learning Classification:**
   - *Objective:* Develop a machine learning classification system to automatically categorize customers into predefined clusters during their initial interaction.
   - *Rationale:* Automating customer categorization at the first visit streamlines marketing efforts and enhances personalized engagement.

3. **Classifier Evaluation:**
   - *Objective:* Systematically evaluate various classifiers using scikit-learn to identify and deploy the most effective model.
   - *Rationale:* Selecting an optimal classifier ensures accurate assignment of clients to their respective categories, enhancing the overall performance of the system.

4. **Confusion Matrix Analysis:**
   - *Objective:* Utilize confusion matrices to assess the performance of the clustering model.
   - *Rationale:* The confusion matrix provides insights into prediction accuracy, particularly valuable in scenarios with potential class imbalances.

5. **Learning Curve Visualization:**
   - *Objective:* Generate learning curves to evaluate model performance, identifying potential issues like overfitting or underfitting.
   - *Rationale:* Learning curves offer a visual representation of the model's behavior, aiding in the assessment of its generalization capabilities.

6. **Ensemble Method Implementation:**
   - *Objective:* Implement ensemble methods, combining multiple classifiers to enhance overall predictive performance.
   - *Rationale:* Ensemble methods often outperform individual classifiers, providing a robust and accurate solution.

7. **Prediction System Development:**
   - *Objective:* Develop a prediction system that leverages the trained machine learning model to categorize customers in real-time.
   - *Rationale:* A functional prediction system facilitates the seamless integration of the developed model into practical business operations.

8. **Performance Comparison Across Multiple Algorithms:**
   - *Objective:* Compare the performance of various machine learning algorithms, including Support Vector Machine, Logistic Regression, k-Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, and Gradient Boosting.
   - *Rationale:* Identifying the most effective algorithm ensures the deployment of a robust and accurate customer classification system.

## Data Processing

### 1. Feature Engineering:
   - **InvoiceDate:**
     - Converted the `InvoiceDate` column to datetime format for temporal analysis.
     - Extracted relevant temporal features such as day, month, and hour for deeper insights.

### 2. Customer Segmentation:
   - Utilized the KMeans clustering algorithm for customer segmentation based on their purchasing behavior.
   - Created 11 customer segments (clusters) to classify customers into distinct categories.

### 3. Feature Creation:
   - Introduced new features to capture essential information:
     - `Basket Price`: Calculated as the product of `Quantity` and `UnitPrice` to represent the total cost of each product in a transaction.
     - `categ_0`, `categ_1`, `categ_2`, `categ_3`, `categ_4`: Percentage of total `Basket Price` contributed by each category.

### 4. Data Aggregation:
   - Aggregated transaction data for each customer based on the introduced features.
   - Summarized customer-specific information, including counts, minimum, maximum, mean, and sum of transaction attributes.

### 5. Correcting Time Range:
   - Adjusted the time range by multiplying `count` and `sum` by a factor of 5 to align with the desired time frame.

### 6. Scaling:
   - Applied Min-Max scaling to the dataset, ensuring that all features are on a comparable scale.

### 7. Feature Selection:
   - Selected a subset of features (`'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4'`) for model training.

### Data Encoding:

   - The dataset primarily consists of numerical features, and no explicit data encoding was necessary for the chosen machine learning models.

These preprocessing steps aimed to enhance the dataset's quality, create meaningful features, and align the data with the project's objectives, facilitating effective machine learning model training and evaluation.

## Algorithms Used

### 1. Support Vector Machine (SVM)

The project begins with the Support Vector Classifier (SVC). Hyperparameter tuning is conducted using grid search to enhance classifier performance.

### 2. Logistic Regression

Utilizing logistic regression, the project explores hyperparameter tuning for optimal model performance.

### 3. k-Nearest Neighbors (kNN)

The kNN classifier is employed with grid search to find the optimal number of neighbors for enhanced performance.

### 4. Decision Tree

Decision tree classification is implemented with hyperparameter tuning to improve accuracy.

### 5. Random Forest

Random Forest classifier is utilized with grid search to find the optimal combination of parameters for enhanced precision.

### 6. AdaBoost Classifier

AdaBoost classifier is explored with hyperparameter tuning to enhance its performance in customer classification.

### 7. Gradient Boosting Classifier

The Gradient Boosting classifier is employed with grid search for optimal hyperparameter selection.

## Ensemble Method

The project employs ensemble methods to combine the predictions of multiple base classifiers to improve overall accuracy and robustness.

## Project Conclusion

The VotingClassifier strategy proved effective in leveraging the diverse strengths of individual classifiers, mitigating the weaknesses inherent in any single algorithm. This amalgamation of predictions from multiple models resulted in a more robust and reliable categorization of customers, enhancing the overall accuracy of the classification task.

However, it is essential to recognize the limitations of the project. Firstly, the efficacy of the classifiers heavily relies on the quality and representativeness of the training data. Any biases or anomalies present in the data could impact the model's generalizability to real-world scenarios. For instance, if the training data predominantly consists of a specific demographic or time period, the model might not perform optimally when faced with diverse customer groups or evolving market trends.

Additionally, the predictive performance may be affected by the choice of features and the assumptions underlying the classifiers. If certain critical features defining customer behavior are omitted or inadequately represented, the model's ability to accurately categorize customers may be compromised.

Furthermore, the precision scores obtained, while indicating a satisfactory level of accuracy, should be interpreted in the context of the specific application. In some scenarios, a precision of 76.75% may be considered acceptable, while in others, particularly those involving high-stakes decision-making, a higher precision may be imperative.

To improve the project, it would be beneficial to conduct a more extensive exploration of feature engineering and selection techniques, ensuring that the chosen features truly capture the nuances of customer behavior. Additionally, incorporating more advanced ensemble methods or exploring neural network architectures may yield further improvements.

In a real-world context, the predictions of customer categorization could significantly impact business strategies. For example, in e-commerce, accurate customer classification could inform targeted marketing campaigns, personalized product recommendations, and efficient allocation of resources for customer retention. Conversely, inaccurate predictions might lead to misdirected marketing efforts, suboptimal resource allocation, and diminished customer satisfaction.

## License

This project is licensed under the [MIT License](LICENSE).

