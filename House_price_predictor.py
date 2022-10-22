# House Price Predictor
import pandas as pd

housing = pd.read_csv('data.csv')
housing.head()
housing.info()
housing['CHAS'].value_counts()
housing.describe()

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# housing.hist(bins=50,figsize=(20,15))
#train-test splitting
import numpy as np

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled  = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]  

train_set,test_set = split_train_test(housing,0.2)
print(f'Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}')

from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f'Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}')

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits =1,test_size =0.2,random_state =42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set.info()
strat_test_set.describe()

strat_test_set['CHAS'].value_counts()

strat_train_set['CHAS'].value_counts()

# # Looking for co-relations

corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending = False)

# %%
# from pandas.plotting import scatter_matrix
# attributes = ['MEDV','RM','ZN','LSTAT']
# scatter_matrix(housing[attributes],figsize=(12,8))

# housing.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)

# ## Attribute Combinations

housing['TAXRM'] = housing['TAX']/housing['RM']
corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending = False)

housing = strat_train_set.drop('MEDV',axis=1)
housing_labels = strat_train_set['MEDV'].copy()

# ## handling missing attribute

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)

print(imputer.statistics_.shape)

X = imputer.transform(housing)
housing_tr = pd.DataFrame(X,columns= housing.columns)
housing_tr.describe()

# # Scikit-Learn Design

# Primarily, three types of objects are
# 1. Estimators - It estimates some parameter based on a dataset. eg. Imputer 
# 2. Transformers - takes input and returns output based on the learnings from fit()
# 3. Predictors - LinearRegression model is an example of predictor. fit() function and predict() are two functions.It also gives 
#                 score() function to evaluate the predictions. 

# # Features of Scaling
# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization) 
#     (value - min)/(max - min)
#     Sklearn provides a class called MinMaxScaler for this.
# 2. Standarization 
#     (value - mean)/std
#     Sklearn provides a class called StandardScaler for this.    

# # Creating a pipeline

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
   ('imputer',SimpleImputer(strategy='median')),
   ('std_scaler',StandardScaler()) 
])

housing_num_tr = my_pipeline.fit_transform(housing)
print(housing_num_tr)

# # Selecting a desired model for price prediction

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
model = RandomForestRegressor()
# model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(list(some_labels))

# # Evaluate the model

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)
print(mse)

# # Use better evaluation technique - Cross Validation

# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring = "neg_mean_squared_error",cv=10)
rmse_score = np.sqrt(-scores)
print(rmse_score)

def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("standard deviation: ",scores.std())

print_scores(rmse_score)

from joblib import dump,load
dump(model,'HPPM.joblib')

# # Testing the model on test data

X_test = strat_test_set.drop('MEDV',axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_text_prepared = my_pipeline.transform(X_test)
final_prediction = model.predict(X_text_prepared)
final_mse = mean_squared_error(Y_test,final_prediction)
final_rmse = np.sqrt(final_mse)
print(final_rmse)



