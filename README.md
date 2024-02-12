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

My goal was to optimize my model to achieve a target predictive accuracy higher than 75%.

# 1st Attempt

      - For the 'APPLICATION_TYPE' column, I chose a cut-off value of 156.
      - For the 'CLASSIFICATION' column, I chose a cut-off value of 1883.
      - 2 Layers:
            - hidden_nodes_layer1 = 25
            - hidden_nodes_layer2 = 10
      - Epoch = 100

![03 Step2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/472fa72e-826b-4511-9e14-059e49019f06)


# 2nd Attempt
      
      - For the 'APPLICATION_TYPE' column, I chose a cut-off value of 528.
      - For the 'CLASSIFICATION' column, I chose a cut-off value of 777.
      - 3 Layers:
            - hidden_nodes_layer1 = 40
            - hidden_nodes_layer2 = 20
            - hidden_nodes_layer3 = 10
      - Epoch = 100

![03 Step2 Optimization1](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/9d36d7e6-0846-45ff-a0a9-fbf233b589c2)


# 3rd Attempt

      - For the 'APPLICATION_TYPE' column, I chose a cut-off value of 156.
      - For the 'CLASSIFICATION' column, I chose a cut-off value of 1883.
      - 3 Layers:
            - hidden_nodes_layer1 = 60
            - hidden_nodes_layer2 = 20
            - hidden_nodes_layer3 = 5
      - Epoch = 75

![03 Step2 Optimization2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/04045b44-1d89-4ba8-908e-0ac61203431b)

            
Some methods I used to optimize my model:

        - Created more bins for rare occurrences in columns.
        
        - Increased/Decreased the number of values for each bin.
        
        - Added more neurons to a hidden layer.
        
        - Added more hidden layers.

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

