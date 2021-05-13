# CHICAGO TAXI TRIP DATA SET MACHINE LEARNING USE CASE
# _README_
# 1. Project Overview

The emergence and rapid rise of app-based, on-demand ride services provided by transportation network companies (TNCs) such as Uber and Lyft, are disrupting the transportation sector and changing how people travel. By introducing a new form of mobility featured with many advantages such as on-demand, low cost (compared to traditional taxi services), and reliability, major TNCs have quickly attracted millions of users and generated billions of trips. As TNCs become increasingly popular, they affect cities both in positive and negative ways. For ex-ample, while many cherish the convenience and economic benefits brought by TNCs to both riders and drivers , others are concerned that TNCs may worsen traffic congestion and threaten the already struggling transit industry. To maximize the potential benefits and to mitigate the harms brought by this new mobility option, it is important for policymakers to understand and forecast the demand for traditional taxi services so that they can plan accordingly and develop regulatory measures.The increasing popularity of machine learning may provide boost to the know the demand patterns of the rides .

Hence we zeroed down this final use case:
_To predict the geographic location with the highest rider density  as a way to optimize the active time of a driver. This can be used to safe fuel expenses , provide cabs at places with high demand ensuing profitability , better traffic management and potentially reduce the customer wait time._

#### GCP Project ID
````
us-gcp-ame-con-01e-npd-1
````
## 2. Platform 
- GCP Big Query 
- AI Platform Notebook
- Google Storage Bucket

## 3. Coding 
- Python 3.8
- Standard SQL

## 4. Libraries
- Numpy 
- Pandas 
- Matplotlib 
- Seaborn 
- Scikitplot
- TensorFlow
- Keras
- Scikitlearn 

## 5. Preprocessing

After extracting the Chicago Taxi Public data set in Big query we ran our sql queries. To check our all SQL Queries, you can refer to the following file.
```
gcp_ml_bigquery.sql
``` 
#### Dataset Id
```
bigquery-public-data:chicago_taxi_trips.taxi_trips
```
Here is one of the sample query:

```
----------------------------CHICAGO DAILY WEATHER DATA----------------------------
--Daily summaries of precipitation, snow and temperature for Chicago were downloaded from 'https://www.ncdc.noaa.gov/cdo-web/search'
--and the csv was uploaded to `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily`, calculated averages for each datapoint over all the Chicago stations
CREATE OR REPLACE TABLE
  `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily_averages` AS (
  SELECT
    DATE,
    ROUND(AVG(prcp),2) AS avg_prcp,
    ROUND(AVG(snow),2) AS avg_snow,
    ROUND(AVG(CAST(tavg AS FLOAT64)),2) AS avg_tavg
  FROM
    `us-gcp-ame-con-01e-npd-1.chcg_dtst.weather_daily`
  WHERE
    EXTRACT(Year
    FROM
      DATE) >2015
  GROUP BY
    DATE
  ORDER BY
    DATE )
```
## 6. Data Visualisation

After cleaning up the data we integreted our jupyter notebook hosted on an VM instance with the big query API and did some data visualisation.
The code to connect to the big query is given below. After the connection you are free to use an queries and python libraries.

```
# Trip Cost Plot
trip_total = """
                SELECT trip_total, count(*)
                FROM `us-gcp-ame-con-01e-npd-1.chcg_dtst.tx_trps_eda`
                GROUP BY trip_total
                ORDER BY trip_total 
"""
# Setting up big query job
safe_config = bigquery.QueryJobConfig()

trip_total_job = client.query(trip_total,job_config=safe_config )

# API request - run the query, and return a pandas DataFrame
trip_total_job_result = trip_total_job.to_dataframe()

#Fetching the results
trip_total_job_result
``` 

we did an analysis for our few of our sub hypothesis. They are as follows:

a)	_Time of the Day affects the Demand for taxi trips
b)	Federal Holidays Affect the taxi trip patterns
c)	Weather and Crime Rate affect demand for taxi trips_

You can refer to this visualisation notebook for more details.

```
ExplorataryDataAnalysis.ipynb
``` 
## 7. Methadology 

The model uses a Sequential Model architecture which is the basic deep learning-based framework for making Multi-Layer Perceptron(MLP) models. The MLP are the basic units of Neural Networks which mimic the behavior of a human brain.
A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation).Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer. 
An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable. 
A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

![image](https://user-images.githubusercontent.com/81349521/118134473-97480f00-b41f-11eb-96c5-b93ef0e3b8f1.png)
                               


 A brief explanation of the model architecture:
- A dense feature layer stores meta data about the features being sent into the model. This helps automate the process of normalization, one-hot encoding, etc without the need of an additional preprocessing pipeline. Hence, the data preprocessing is done partly as part of the model input step.

- The dense feature layer is directly sent to a MLP with appropriate Activation functions. On the last layer, we use the SoftMax activation function since this we are solving a multiclassification problem.

# 8.Setup

The following steps are required, regardless of your notebook environment.

##### a) Define your configuration Variables 

The following steps are required, regardless of your notebook environment.

This is the process of providing static configuration details such as the Project Name, Bucket Name etc along with some variable details such as the Number of Epochs etc. The static details can be changed to work with any other GCP project while the Variable details can be used to run HyperParameter Tuning jobs.

snippet of our code is below 

```

# Use project ID from console page
!gcloud config set project us-gcp-ame-con-01e-npd-1

#project name
PROJECT_NAME = "us-gcp-ame-con-01e-npd-1"

# Shows up in models pane of AI platform, helps you track your key project
MODEL_NAME = "Experiment_One"

# This defines your each experiment, try to keep it meaningful
MODEL_VERSION = "v2"

#region
REGION = "us-east4"

# JOB_NAME is auto-incremental. It will be appended by the latest timestamp when submitting the training job.
JOB_NAME = MODEL_NAME + '_' + MODEL_VERSION

#Google Storage Bucket
BUCKET_NAME = "us-gcp-ame-con-01e-npd-1-modelartifacts"

#sub folder name 
SUB_FOLDER_NAME = MODEL_NAME + "/" + MODEL_VERSION

# Creates gs directory with this name - persists transient files during AI Platform jobs tun
JOB_DIR = 'gs://' + BUCKET_NAME + '/' + SUB_FOLDER_NAME

# Should be same as JOB_DIR with model_save as folder added
MODEL_DIR = JOB_DIR + "/model_save/"

#The number of times the data is shown to the Model. This is a hyper parameter and can be tuned
NUM_EPOCHS = 114

#The restriction placed on the dataset to retrive our train data. 
TRAIN_RESTRICTION = "'trip_start_timestamp <= 1577836799'"
#1546300800
#The restriction placed on the dataset to retrieve our validation data. Since this is only a training notebook, it has no Evaluation
EVAL_RESTRICTION = "'trip_start_timestamp >{}  and trip_start_timestamp<={}'"

#1546300800 and trip_start_timestamp<=1567276140
#Batch size for training
TRAIN_BATCH_SIZE = 512


``` 

#### b) Create a skeleton for the Trainer
According to the AI Platform documentation the training code has to be modularised into a Trainer package with a defined layout. The following steps create that layout. Here is the code snippet below:

```
import os, shutil

PACKAGE_NAME = 'experiment1'

MODEL_FILE_PATH = '{}'.format(PACKAGE_NAME)

if os.path.isdir(MODEL_FILE_PATH):
    
    shutil.rmtree(MODEL_FILE_PATH)
    
!mkdir $MODEL_FILE_PATH

trainer_path = MODEL_FILE_PATH + '/trainer.py'

init_path = MODEL_FILE_PATH + '/__init__.py'
``` 



#### c) Create the Setup File

The Setup File is required to download any additional package that was used during the Modeling process. This is placed outside the Trainer model. The code snippet is below:

```
SETUP_FILENAME =  "setup.py"

print("setup filename: " + SETUP_FILENAME)

%%writefile $SETUP_FILENAME

# Setup file to install necessary packages
from setuptools import find_packages
from setuptools import setup

# You can define all the packages you need for your job below along with the versions

REQUIRED_PACKAGES = ['tensorflow-io==0.11.0','fastavro']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
``` 
#### d) Creating the training module

This is the Driver program of the entire module. It contains the functions to generate the Train and Evaluation Datasets along with the design of the model architecture and Model Training steps. You can create the model according to your preference.

Check this python notebook for more details:
```
Training-Experiment-1.ipynb
Training-Experiment-1.ipynb
Training-Experiment-2.ipynb
Training-Experiment-3.ipynb
``` 

## 9. Implementation

After doing our data cleaning procedure and our exploratory data analysis we proceeded with our  implementation phase. The implementation procedure was done with the following steps:
1.	Data Preparation
2.	Model Training
3.	Model Evaluation and Hyper-parameter Tuning
4.	Model Deployment and Inference


#### 9.1 Data Preparation

The process of generating transforming the data to be model ready is referred to as the Data Preparation. The following steps were taken for preparing the data :

Step 1 : The data was split into train, validation and test sets. This was done by splitting the data along the timestamp.
Step 2 : For the training set, all the data before January 2021 was filtered out. 
Step 3 : The data was biased against the Central location. This was because majority of the data points collected were from the busiest part of Chicago(Central). Hence, the data had to be under sampled to create a holistic model. This was done by finding the region with the least records and randomly selecting same number of records from the other regions.
Step 4 : The training set was further split into two parts. The first one did not contain the data for the year 2020 while the second one contained the entire training set. This was done to verify if the year with COVID-19 had any impact on the model development process.
Step 5 : The validation data was created from the training set. The validation set selected was from the 1st of January 2019 to August 31st 2019. This was only used to calculate the hyper parameters. The model was retrained on the entire train set before generating the predictions.

#### 9.2	Model Training

##### Step 1: Define the model architecture
For building the Geographic Locations classifier we create a Sequential model. The baseline architecture of this model is present in TensorFlow Keras Library.
The model itself works on the principles of a simple single layer Neural Network. Neural Networks are a class of soft computing algorithms used to mimic the human brain. The Neural Network algorithms work by passing numeric inputs into a hierarchical structure of mathematical functions and reducing the error generated at the end of the hierarchy.

In this experiment, we use this model architecture for generating a feature layer and predicting the geographic cluster

##### Step 2: Create the input and output layers for the model and use the appropriate Hyper   Parameters 
The Sequential model has multiple Hyperparameters like the number of epochs, learning rate of the model, beta values, epsilon, batch size, etc.

These hyperparameters help us build a model that converges to ground source of truth (i.e. recommendations) in the fastest and most accurate manner.

Note: As a separate exercise we also run Hyperparameter tuning jobs on AI Platform to find the most optimal model hyperparameters .The input layers and output layers are defined according to the problem statement. 

In our current experiments, we are using 12 input features (except for Experiment 3 which is a real time prediction model) and predicting 9 classes. This is a very important step as this defines the format of the inputs to be sent and outputs received from the model while inferring.

##### Step 3: Split the preprocessed data into Train, Validation and Test datasets
For this step, we do not follow the tradition percentage split but instead, split the data according to the time stamp. 
Following the above 3 steps we ran 3 experiments which varied in different train data set , feature selection and hyperparameters. In the section below, we explain each experiment in detail.


##### Experiment One
a.	Since the Chicago Dataset is inherently imbalanced towards the Central location, we only train our model from the under sampled dataset. For this, we pick a time range from the start of 2016 to the start of 2019(i.e. Jan 01 2019). 
b.	For the Hyper Parameter tuning, we pick the validation dataset from the 1st of Jan 2019 to the 31st of August 2019.
c.	Based on the Hyper Parameter values, the model is retrained on the entire train dataset from the start of 2016 to the end of 2019.
d.	This is then used to generate predictions on the test set which is for the months of Jan 2020 and Jan 2021

##### Experiment Two
a.    For this experiment, we pick a time range from the start of 2016 to the start of 2020
        (i.e.   Jan 1 2020). 
b.	For the Hyper Parameter tuning, we pick the validation dataset from the 1st of Jan 2020 to the 31st of August 2020. [Since the architecture is the same, we did not run a Hyper Parameter Job for this model]
c.	Based on the Hyper Parameter values, the model is retrained on the entire train dataset from the start of 2016 to the end of 2020.
d.	This is then used to generate predictions on the test set which is for the month of Jan 2021. We don’t predict for Jan 2020 since that is part of the training set for this experiment

##### Experiment Three
Experiment three is exactly like Experiment One with a small change in the feature set. For Experiment Three, we use features that are readily available as supposed to features that are generated while on a trip like trip miles , trip total and payment type.

For more details on the experiments, hyperparamter tuning and the code related to it. You can refer to these following notebooks.
```
Training-Experiment-1.ipynb
Training-Experiment-2.ipynb
Training-Experiment-3.ipynb
HyperParameterTuning.ipynb
``` 
### 10.	Model Evaluation
This section introduces the various metrics used to evaluate the performance of the model. Since the model performs binary classification, precision, recall, and f1 score are used as evaluation metrics along with the overall accuracy. Description for each metric is provided as follows:

##### ROC AUC Score
ROC AUC score (Area under the ROC Curve) roughly translates to the ability of the model to distinguish between the two labels.

##### Precision: 
It is defined as the total number of correct predictions divided by the total number of predictions for each class. Precision can be mathematically expressed as TP / (TP+FP)

##### Recall:
It is defined as the total number of correct predictions divided by the total number of actual values for each class. Mathematically expressed as TP / (TP+FN) 

Cohen Kappa Score: Cohen’s kappa is defined as:
```
k=(p_o-p_e)/(1-p_e )
``` 

where po is the observed agreement, and pe is the expected agreement. It basically tells you  
how much better your classifier is performing over the performance of a classifier that simply 
guesses at random according to the frequency of each class.

For each model created using experiments, we ran it on our respective test data plotted the results. We did in two ways:
1. Prediction in Memory - load the model in the notebook and run it.
2. AI Platform Prediction - create a structure for the new incoming data so that it can be fed into the model. The model will then run as a job in AI platform and then submit the output in a cloud storage bucket.

![ExperimentOneTable](https://user-images.githubusercontent.com/81349521/118136423-bb0c5480-b421-11eb-9a73-0df12d792424.PNG)


All the related code is in the following notebook :
```
Inference.ipynb
``` 

### 11. Inference 

Running all the experiments and checking their results, we have a number of findings to talks about. Since from our report we can see our test data is imbalanced in all three experiments , we have used the ROC AUC Score, Cohen Kappa Score . Also, calculating the accuracy of the of the model could be misleading since it will not reflect prediction performance of smaller classes (shadowed by performance of any much bigger class) . 
Hence, we will instead analyze and focus more on the precision , recall , F1 and macro average metrics. 

Firstly, Just by looking at the ROC AUC Score and Cohen Kappa Score we see that Experiment 1 performs the best . Experiment third comes in the second place and the Experiment third performs the worst among all three. 

Digging deeper we see that that data is highly skewed towards the classes 0,1 and 8. Checking their F1 scores , precision recall scores, these classes have been classified well. What is more interesting is how the smaller classes like 2,3,5 are performing. In our use case , we would give more weightage to the  recall score than the precision score, since even if the prediction is incorrect or not precise , the taxi can go to some other location and still earn revenue or cater to the taxi demand , whereas we don’t want many taxis to lose out on zones which have been historically demanded by the users. Therefore we feel recall is a better and important indictor for us . 

Comparing their Recall score . we see experiment 1 outperforms experiment 2 and 3 for the small classes except that class 2 gives a higher recall for experiment 2 than 1.But, we notice that it also drops out the scores of the class 1 drastically. Also, it is important to notice that the training data of experiment 2 also includes the data which was captured during the COVID-19 pandemic which could act as an anomaly for a trend or a pattern. We experimented with this data too because we wanted to check if these unusual period of data affects the model or not. Seeing the results , we don’t see any improvements. Instead, we see worse result in almost all cases and hence it would be fair to write off this particular model.
Also a good measure would be to see the macro average of each indicator so that we give equal weights to all the classes, and as expected we see that experiment 1 is better in every metric.

### 12. Conclusion

Hence, based on the three experiments, the Model created from experiment one can be used to predict the regions accurately post the completion of a trip. Though this is limited in scope for a real time application, this model can be used to direct cabs based on the time of the day. For a real-time implementation, we can use the model created from experiment three. The Experiment three model has a few shot comings which will have to be dealt using a rule-based approach. The third model fails to predict regions which are not visited frequently. Hence, for the zones which are rarely visited, the recommendations must be made with an undertone of caution of failure. This model can be retrained once we get more data points on the less frequently visited regions to holistically suggest a region in real-time. 



