# 21-Deep-Learning

## Step 1: Preprocess the Data

- Preprocessed the dataset using Pandas and scikit-learn’s StandardScaler()

- Read in the charity_data.csv to a Pandas DataFrame.

    TARGET for my model: 'IS_SUCCESSFUL' column from application_df
    FEATURES for my model: Every other column from the application_df (dropped the 'IS_SUCCESSFUL' column to create the features data frame)

- The 'EIN' and 'NAME' columns were dropped since they were neither targets nor features for my model.

- I used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then checked if the binning was successful.
      For the 'APPLICATION_TYPE' column, I chose a cut-off value of 156.
      For the 'CLASSIFICATION' column, I chose a cut-off value of 1883.

- I used pd.get_dummies() to encode categorical variables.

- I split the preprocessed data into a features array, X, and a target array, y. I used these arrays and the train_test_split function to split the data into training and testing datasets.

- I scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, and then using the transform function.

## Step 2: Compile, Train, and Evaluate the Model

Using my knowledge of TensorFlow, I designed a neural network, and deep learning model, to create a binary classification model that predicted if the Alphabet Soup-funded organization would be successful based on the features in the dataset. I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

- I created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
    input features = length of the Scaled Training Data
    hidden_nodes_layer1 = 25
    hidden_nodes_layer2 = 10

- The structure of the model:
  
  ![01 Step2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/ca01f51c-c586-43b6-8636-3b9f56cfd6d5)

- I compiled and trained the model.

  ![02 Step2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/0ecc4da1-2502-443b-8003-045f216cd5d8)

- I evaluated the model using the test data to determine the loss and accuracy.

  ![03 Step2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/aa9ccf82-eb67-4019-a38e-cda426f00733)

- I saved and exported your results to an HDF5 file and named the file AlphabetSoupCharity.h5.

## Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

    Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
        Dropping more or fewer columns.
        Creating more bins for rare occurrences in columns.
        Increasing or decreasing the number of values for each bin.
        Add more neurons to a hidden layer.
        Add more hidden layers.
        Use different activation functions for the hidden layers.
        Add or reduce the number of epochs to the training regimen.

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

    Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

    Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

    Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

    Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

    Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

## Step 4: Write a Report on the Neural Network Model

# Overview of the analysis: 
    Explain the purpose of this analysis.

# Results: 
    Using bulleted lists and images to support your answers, address the following questions:

# Data Preprocessing
    What variable(s) are the target(s) for your model?
    What variable(s) are the features for your model?
    What variable(s) should be removed from the input data because they are neither targets nor features?

# Compiling, Training, and Evaluating the Model
    How many neurons, layers, and activation functions did you select for your neural network model, and why?
    Were you able to achieve the target model performance?
    What steps did you take in your attempts to increase model performance?

# Summary: 
    Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

