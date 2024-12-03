# **Sign Language Recognition with Deep Learning**

This project implements a convolutional neural network (CNN) to recognize American Sign Language (ASL) letters using the Sign Language MNIST dataset. It includes data preprocessing, model training, evaluation, and visualization of results.

## **Project Overview**
This project aims to:
- Build a CNN model capable of recognizing ASL letters.
- Use data augmentation to improve generalization.
- Evaluate the model's performance with metrics like accuracy and confusion matrix.

## **Dataset**
The dataset used in this project is the [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist). It consists of:
- **Training set:** 27,455 grayscale images of 28x28 pixels with corresponding labels.
- **Test set:** 7,172 grayscale images of 28x28 pixels with corresponding labels.
- The labels represent letters A-Z, excluding J.

## **Requirements**
The project uses the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `scikit-learn`

To install these dependencies:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

## **Steps to Run the Project**
1. Load and Prepare the Data
Load the training and test datasets using pandas.
Reshape the images into a 4D tensor of shape (samples, 28, 28, 1).

2. Visualize the Data
Display a grid of sample images from the training dataset along with their labels.

3. Data Augmentation
Use ImageDataGenerator to augment the training data with transformations such as: Rotation

4. Build the Model
The CNN architecture includes:
Three convolutional layers with ReLU activation, batch normalization, and max pooling.
A dense layer with ReLU activation and a dropout rate of 0.5 for regularization.
A final dense layer with 26 neurons (softmax activation) for classification.

5. Train the Model
Compile the model using the Adam optimizer with a learning rate of 0.001.
Train the model for 20 epochs with a batch size of 32.

6. Evaluate the Model
Evaluate the model using: Accuracy, Loss, A confusion matrix and a classification report.

7. Visualize Training Results
Plot training and validation accuracy/loss to analyze model performance.

## **Results**
- Achieved a high accuracy on both training and test datasets.
- Generated a detailed classification report with precision, recall, and F1 scores.
- Visualized the confusion matrix to understand model performance across classes.

## **How to Run**
Clone the repository:
```bash
   git clone https://github.com/csm34/SignLanguage.git
   cd SignLanguage
```

## **License**
This project is licensed under the MIT License. See LICENSE for more information.

