## Task 1: Logistic regression

### Step 1: Data Preprocessing

#### 1.1 Loading Dataset:
- The dataset was loaded from a CSV file using the Pandas library.

#### 1.2 Target Variable Transformation:
- To perform binary classification, the target variable was converted into numerical values.
- Since it's binary classification, the target classes were defined as follows:
  - 'Setosa' was encoded as 1.
  - Other species, referred to as 'Not Setosa', were encoded as 0.

#### 1.3 Data Missing values Check:
- After the transformation of the target variable, a check was performed to ensure that the dataset contained no missing values. This step is crucial to ensure the quality of the dataset.

#### 1.4 Data Splitting:
- Following the data integrity check, the dataset was split into training and testing sets.
- The split was performed with an 80% portion of the data allocated for training and the remaining 20% for testing. This ratio is a common practice in machine learning to balance model training and evaluation.

### Step 2: Training

#### 2.1 Initialization of Parameters:
- The logistic regression model requires two main parameters: weights (w) and bias (b).
- Weights (w) are initialized with random values. 
- The bias (b) is initialized to 0.

#### 2.2 Training Loop:
- The training loop iterates over a fixed number of epochs. An epoch represents a single pass through the entire training dataset.
- In each epoch:
  - **Forward Pass:**
    - The forward pass computes the linear combination of features and weights.
    - The result is then passed through the sigmoid function to obtain the predicted probabilities of each sample belonging to the positive class.
  - **Loss Computation:**
    - The loss function used is binary cross-entropy. This function quantifies the difference between the predicted probabilities and the actual labels.
    - Loss = -np.mean(y_train * np.log(a) + (1 - y_train) * np.log(1 - a))
  - **Backpropagation:**
    - Backpropagation computes the gradients of the loss function with respect to the model parameters (w and b).
    - dw = np.dot(X_train.T, (a - y_train)) / len(X_train)  # for the weights.
    - db = np.mean(a - y_train) # for the bias
  - **Parameter Update:**
    - The model parameters (w and b) are updated using gradient descent.
    - w -= lr * dw  # for the weights, where lr is the learning rate.
    - b -= lr * db  # for the bias, where lr is the learning rate.
- The loop continues for the specified number of epochs, gradually optimizing the model parameters to minimize the loss function and improve predictive accuracy.

### Step 3: Making Predictions and Model Evaluation:
- After the model has been trained using the training set, it is used to make predictions on the testing set.
- The updated weights (w) and bias (b) values obtained from the training process are utilized.
- The predictions are made by passing the features of the testing set through the trained model:
  - The linear combination of features and weights, plus the bias 
  - The sigmoid function is applied to the result to obtain predicted probabilities.
  - If the predicted probability is greater than 0.5, the model predicts the sample as 'Setosa'; otherwise, it predicts it as 'Not Setosa'.
- After prediction we will compare the true labels with the predicted labels to calculate the accuracy to evaluate the model performance.

### Results: 
- Model Accuracy calculated as 95.24%
  
![image](https://github.com/Hareb4/ai-datathon/assets/160310286/411a2df9-578f-4864-a12b-a7c99a90ca25)



### Code :
- Colab Link : https://colab.research.google.com/drive/15qEke19BfR3k7A-fMScoRF2gvJzq5tzt?usp=sharing
