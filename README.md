# Insurance_Premium_Prediction

## Problem Statement :
The goal of this project to give people an estimate of how much they need based on their individual health situation. After that, customers can work with any health insurance carrier and its plans and perks whilwe keeping the projected cost from our study in mind. This can assist a person in concentrating on the health side of an insurance policy rather than the ineffective part.

## Dataset :
The dataset is taken from a Kaggle. You can download the dataset from [here](https://www.kaggle.com/noordeen/insurance-premium-prediction)

## Approach :
Applying machine learing tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and model testing to build a solution that should able to predict the premium of the personal for health insurance.

- **Data Exploration :** Exploring the dataset using pandas, numpy, matplotlib, plotly and seaborn.
- **Exploratory Data Analysis :** Plotted different graphs to get more insights about dependent and independent variables/features.
- **Feature Engineering :** Numerical features scaled down and Categorical features encoded.
- **Model Building :** In this step, first dataset Splitting is done. After that model is trained on different Machine Learning Algorithms such as:
    1) Linear Regression
    2) Decision Tree Regressor
    3) Random Forest Regressor
    4) Gradient Boosting Regressor
 
- **Model Selection :** Tested all the models to check the R-squared and best R-squared model selected for GridSearchCV and best parameters to be find out.
- **Pickle File** : Selected model as per best R-squared and GridSearchCv best parameters,and created pickle file using pickle library.
- **Webpage &Deployment :** Created a web application that takes all the necessary inputs from the user & shows the output. Then deployed project on the Heroku Platform.

## Deployment Link :
Heroku Deployment

https://insurance--prediction.herokuapp.com/

## Web Inerface :
![alt text](https://github.com/DeepakRautella/Insurance_Premium_Prediction/blob/main/images/home.png)

![alt text](https://github.com/DeepakRautella/Insurance_Premium_Prediction/blob/main/images/trained.png)

![alt text](https://github.com/DeepakRautella/Insurance_Premium_Prediction/blob/main/images/experiment.png)

![alt text](https://github.com/DeepakRautella/Insurance_Premium_Prediction/blob/main/images/predict.png)

![alt text](https://github.com/DeepakRautella/Insurance_Premium_Prediction/blob/main/images/log.png)

![alt text](https://github.com/DeepakRautella/Insurance_Premium_Prediction/blob/main/images/artifact.png)

## Libraries used :
    1) Pandas
    2) Numpy
    3) Matplotlib, Seaborn, Plotly
    4) Scikit-Learn
    5) Flask
    6) HTML
    7) CSS


## Technical Aspects :
    1) Python 
    2) Front-end : HTML, CSS
    3) Back-end : Flask
    4) Deployment : Heruko
