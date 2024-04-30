# Importing the basic libraries
import numpy as np  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_log_error
import warnings
# Disable all warnings
warnings.filterwarnings ('ignore')

train = pd.read_csv('playground-series-s4e4/train.csv')
train.head()
train.info()
train.isnull().sum()

df = train.copy(deep='True')
df = df.drop(columns='id',axis=1)
df.head()


numeric_features = df.select_dtypes(include = ['int', 'float']).columns.to_list()
categoric_features = df.select_dtypes(include = ['object', 'category']).columns.to_list()

# Feature Processing
test = pd.read_csv('playground-series-s4e4/test.csv')
test.head()


ordinal = OrdinalEncoder(dtype='float')
train[categoric_features] = ordinal.fit_transform(train[categoric_features])
test[categoric_features] = ordinal.transform(test[categoric_features])


# Model Training
test = test.drop(columns=['id'], axis=1)
train = train.drop(columns=['id'], axis=1)


y = train.Rings
X = train.drop(['Rings'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)

xgb_params = {
    'verbosity': 0,
    "early_stopping_rounds":25,
    'n_estimators': 1500,
    'eval_metric': 'rmse', 
    'random_state':1234,
    'max_depth': 8, 
    'subsample': 0.79, 
    'objective': 'reg:squarederror',
    'learning_rate': 0.013,
    'colsample_bytree': 0.56, 
    "min_child_weight": 10, 
}

xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predictions
pred = xgb_model.predict(X_val)
pred
print("The RMSLE score is: ", mean_squared_log_error(pred, y_val)**0.5)
predictions = xgb_model.predict(test)


# Submission
submission = pd.read_csv('playground-series-s4e4/sample_submission.csv')

submission['Rings'] = predictions
submission.to_csv('submission.csv', index=False)





