# Breast Cancer Prediction using Neural Networks

This project demonstrates the use of a Neural Network to predict whether a breast tumor is malignant or benign using the Breast Cancer dataset from sklearn. The model is built using TensorFlow and Keras, and it performs binary classification.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Making Predictions](#making-predictions)
- [Usage](#usage)
- [License](#license)

## Dataset
The dataset used in this project is the Breast Cancer dataset from sklearn, which contains 30 features about cell nuclei present in breast cancer biopsies. The target variable indicates whether the tumor is:
- `0`: Malignant
- `1`: Benign

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `sklearn`
  - `tensorflow`

### Instructions
1. Clone this repository.
2. Ensure that the necessary libraries are installed.
3. Run the code provided in the `breast_cancer_prediction.py` file.

## Data Preparation

1. **Loading the Data**: The Breast Cancer dataset is loaded from sklearn's datasets.
2. **Creating a DataFrame**: The data is loaded into a pandas DataFrame with the features as columns.
3. **Adding Target Column**: A new column `label` is added to the DataFrame to represent the target variable (malignant or benign).
4. **Data Preprocessing**:
   - Checking for missing values.
   - Statistical analysis of the data.
   - Standardization of the features.

## Model Architecture

A Neural Network is built using TensorFlow and Keras with the following layers:
- **Input Layer**: Flatten layer to convert the input data into a one-dimensional array.
- **Hidden Layer**: Dense layer with 20 neurons and ReLU activation.
- **Output Layer**: Dense layer with 2 neurons and sigmoid activation, representing the two classes (malignant and benign).

## Training the Model

- **Loss Function**: Sparse categorical cross-entropy is used as the loss function.
- **Optimizer**: Adam optimizer is used to minimize the loss function.
- **Metrics**: Accuracy is used as the performance metric.

The model is trained on 90% of the training data, with 10% used for validation over 10 epochs.

## Model Evaluation

After training, the model is evaluated on the test set to determine its accuracy. The model's accuracy, loss, and validation metrics are visualized using Matplotlib.

## Making Predictions

The trained model is used to predict the class of a tumor based on new input data. The prediction is returned as a probability distribution across the two classes, and the argmax function is used to convert this into a class label (0 for malignant, 1 for benign).

## Usage

To use the model for predictions on new data:

```python
input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,
              0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,
              12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape array for a single data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data
input_data_std = scaler.transform(input_data_reshaped)

# Make prediction
prediction = model.predict(input_data_std)
prediction_label = np.argmax(prediction)

if prediction_label == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
