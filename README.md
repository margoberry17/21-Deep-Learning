# 21-Deep-Learning

## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I used the features in the provided dataset to create a binary classifier that could predict whether applicants would be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are several columns that capture metadata about each organization, such as:

    EIN and NAME—Identification columns
    APPLICATION_TYPE—Alphabet Soup application type
    AFFILIATION—Affiliated sector of industry
    CLASSIFICATION—Government organization classification
    USE_CASE—Use case for funding
    ORGANIZATION—Organization type
    STATUS—Active status
    INCOME_AMT—Income classification
    SPECIAL_CONSIDERATIONS—Special considerations for application
    ASK_AMT—Funding amount requested
    IS_SUCCESSFUL—Was the money used effectively


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

### 1st Attempt

      - For the 'APPLICATION_TYPE' column, I chose a cut-off value of 156.
      - For the 'CLASSIFICATION' column, I chose a cut-off value of 1883.
      - 2 Layers:
            - hidden_nodes_layer1 = 25, activation function = ‘relu’
            - hidden_nodes_layer2 = 10, activation function = ‘relu’
      - Epoch = 100

![03 Step2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/472fa72e-826b-4511-9e14-059e49019f06)


### 2nd Attempt
      
      - For the 'APPLICATION_TYPE' column, I chose a cut-off value of 528.
      - For the 'CLASSIFICATION' column, I chose a cut-off value of 777.
      - 3 Layers:
            - hidden_nodes_layer1 = 40, activation function = ‘relu’
            - hidden_nodes_layer2 = 20, activation function = ‘relu’
            - hidden_nodes_layer3 = 10, activation function = ‘relu’
      - Epoch = 100

![03 Step2 Optimization1](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/9d36d7e6-0846-45ff-a0a9-fbf233b589c2)


### 3rd Attempt

      - For the 'APPLICATION_TYPE' column, I chose a cut-off value of 156.
      - For the 'CLASSIFICATION' column, I chose a cut-off value of 1883.
      - 3 Layers:
            - hidden_nodes_layer1 = 60, activation function = ‘relu’
            - hidden_nodes_layer2 = 20, activation function = ‘relu’
            - hidden_nodes_layer3 = 5, activation function = ‘relu’
      - Epoch = 75

![03 Step2 Optimization2](https://github.com/margoberry17/21-Deep-Learning/assets/136475202/04045b44-1d89-4ba8-908e-0ac61203431b)

            
Some methods I used to optimize my model:

        - Created more bins for rare occurrences in columns.
        
        - Increased/Decreased the number of values for each bin.
        
        - Added more neurons to a hidden layer.
        
        - Added more hidden layers.

        - Added/reduced the number of epochs to the training regimen.

## Step 4: Summary

After three attempts, the model was able to reach an accuracy score of 73.8% (third attempt) in predicting whether or not applicants for funding would be successful. After I implemented multiple approaches to optimize my model I continued to get a result of ~73% for all three attempts. I would consider adding 'NAME' or 'EIN' back into the data to see if I removed too much information, potentially adding more layers to filter through the data better, or using different activation functions for the hidden layers and be able to predict and classify information with higher accuracy. 
