## Task 2: FashionMNIST Classification using PyTorch

### Step 1: Data Loading and Preprocessing

#### 1.1 Loading Dataset:
- FashionMNIST dataset was loaded from the torchvision library.
- Transforms including conversion to tensors and normalization were applied during loading.

#### 1.2 Data Splitting:
- The dataset was split into training and testing sets.
- 85% of the data was allocated for training, while the remaining 15% was reserved for testing.

### Step 2: Model Definition and Training

#### 2.1 Neural Network Architecture:
- The neural network model (`Net`) was defined with multiple fully connected layers.
- ReLU activation was used after each layer except the last one.
- The final layer outputs the class probabilities.

#### 2.2 Training Loop:
- The model was trained using cross-entropy loss and the Adam optimizer.
- Training was performed over a fixed number of epochs.
- Mini-batch gradient descent was employed for parameter updates.

### Step 3: Model Evaluation

#### 3.1 Accuracy Calculation:
- The trained model was evaluated on the test set to calculate accuracy.
- Accuracy was computed as the percentage of correctly classified images.

#### 3.2 Confusion Matrix:
- A confusion matrix was generated to analyze classification performance across different classes.

### Step 4: Visualization

#### 4.1 Sample Predictions:
- Sample images from the test set were displayed alongside their ground truth and predicted labels.
- This provides a visual representation of the model's performance.

### Results: 
- Accuracy of the network on the test set was 88%.
- The confusion matrix revealed the distribution of correct and incorrect predictions across different clothing categories.

  ![image](https://github.com/Hareb4/ai-datathon/assets/160310286/49c96141-9803-48fb-b827-537fae4398cd)

  ![image](https://github.com/Hareb4/ai-datathon/assets/160310286/0bb0160f-71e4-40f7-ba96-717c110a1107)




### Code:
- Colab Link: https://colab.research.google.com/drive/1ZjViBwQAPgacM46p-sfE18H0C59kvexG?usp=sharing
