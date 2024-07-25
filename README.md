# Bank-Marketing-Capstone-Project

# Description


1. Title: Bank Marketing

2. Relevant Information:

   The data is related with direct marketing campaigns of a Portuguese banking institution. 
   The marketing campaigns were based on phone calls. Often, more than one contact to the same client    was required, 
   in order to access if the product (bank term deposit) would be (or not) subscribed. 

   The classification goal is to predict if the client will subscribe a term deposit (variable y).


# import the necessary Labr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data

data = pd.read_csv('bank-full.csv')

pd.options.display.max_columns = None

# Display Top 5 Rows of The Dataset

data.head()

# Check Last 5 Rows of the Dataset

data.tail()

# Find Shape of our Data (How Many Number Of Rows And Columns)

data.shape

print("Number of Rows :-",data.shape[0]),
print("Number of Columns :-",data.shape[1])

data.value_counts('y')

# Calculating Summary Statistics

summary_statistic = data.describe()
summary_statistic

data.info()

# Checking Duplicate values

data.duplicated().any()

# Checking Null Values in Dataset

data.isnull().sum()

data.head()

#Histogram Plot
fig, axes=plt.subplots(2,3 , figsize=(18,6))
sns.histplot(data, x='age', ax=axes[0,0])
sns.histplot(data, x='balance', ax=axes[0,1])
sns.histplot(data, x='day', ax=axes[0,2])
sns.histplot(data, x='campaign', ax=axes[1,0])
sns.histplot(data, x='pdays', ax=axes[1,1])
sns.histplot(data, x='previous', ax=axes[1,2])

# Box Plot

fig, axes= plt.subplots(2,3, figsize=(18,6))

sns.boxplot(data, x='age', ax=axes[0,0])
sns.boxplot(data, x='balance', ax=axes[0,1])
sns.boxplot(data, x='day', ax=axes[0,2])
sns.boxplot(data, x='campaign', ax=axes[1,0])
sns.boxplot(data, x='pdays', ax=axes[1,1])
sns.boxplot(data, x='previous', ax=axes[1,2])

# Class Distribution

print(data.y.value_counts())
axes=sns.countplot(x='y', data=data)
plt.title("Bank Deposite")

Observation
1. Bank Deposite Product subscribed by 5289 peoples out of 45211
2. Bank Deposite Product not subscribed by 39922 peoples out of 45211
3. The Data Set is imbalanced data because the Not Subscribed Count is Greater then subscribed 

# Heatmap

plt.figure(figsize=(8,5))
ploting = data.corr(numeric_only=True)
sns.heatmap(ploting, cmap='BrBG', annot=True)


data.shape

data.head()

# Data preprocessing using StandardScaler

from sklearn.preprocessing import StandardScaler, OneHotEncoder

data_num = data.copy()

sc = StandardScaler()
num_cols = ['age', 'balance', 'day', 'pdays', 'campaign','previous']
data_num[num_cols] = sc.fit_transform(data_num[num_cols])
data_num = data_num.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], axis=1)
data_num.head()

# Encode Categorical Features

encoder = OneHotEncoder(sparse=False)

#Copy original dataframe to df_target
df_target = data.copy()
df_target.head()

df_target = df_target.drop(columns=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'])

#Encode target values yes to 1 and no to 0 
df_target['y'] = df_target['y'].apply(lambda x: 1 if x == 'yes' else 0)

df_target.head()

#Encode Categorical features
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
catg_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

df_categ = data.copy()
df_categ.head()

# Encoding Categorical data to numeric data
df_encoded = pd.DataFrame(encoder.fit_transform(df_categ[catg_cols]))

# Getting feature names from encoder.get_feature_names_out
encoded_feature_names = encoder.get_feature_names_out(input_features=catg_cols)

# Assigning the feature names to the encoded DataFrame columns
df_encoded.columns = encoded_feature_names

#Replace Categorial with Encoding data
df_categ = df_categ.drop(catg_cols, axis=1)
df_categ = pd.concat([data_num, df_encoded, df_target], axis=1)

print('Shape of DataFrame : ', df_categ.shape)
df_categ.head()

# Split Dataset for Training and Testing

#Split Data into 2 dataset, Training and Testing, In this We split data into training and testing group with the ratio of 80:20.

# Selecting Features
feature = df_categ.drop('y', axis=1)

#Selecting Target
target = df_categ['y']

# Training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target,
                                                   shuffle = True,
                                                   test_size=0.2,
                                                   random_state=1)

#Showing Training and Testing Data Results

print('Shape of training feature : ', X_train.shape )
print('Shape of testing feature : ', X_test.shape )
print('Shape of training label : ', y_train.shape )
print('Shape of testing label : ', y_test.shape )

# Logistic Regression 

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)

y_pred1 = log.predict(X_test)

from sklearn.metrics import accuracy_score
training_data_test = log.predict(X_train)
accuracy_score(y_train,training_data_test)

accuracy_score(y_test,y_pred1)

from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)


recall_score(y_test,y_pred1) # this is the better option

f1_score(y_test,y_pred1) # this one too

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

# Initialize the DecisionTreeClassifier with max_depth
dt_pruned = DecisionTreeClassifier(max_depth=5)  # You can adjust the max_depth value
# Fit the model
dt_pruned.fit(X_train, y_train)

y_pred_pruned = dt_pruned.predict(X_test)

training_data_test1 = log.predict(X_train)

accuracy_score(y_train,training_data_test1)

accuracy_score(y_test,y_pred_pruned)

precision_score(y_test,y_pred_pruned)

recall_score(y_test,y_pred_pruned)

f1_score(y_test,y_pred_pruned)

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred3 = rf.predict(X_test)

training_data_test2 = log.predict(X_train)

accuracy_score(y_train,training_data_test2)

xaccuracy_score(y_test,y_pred3)

precision_score(y_test,y_pred3)

recall_score(y_test,y_pred3)

f1_score(y_test,y_pred3)

final_data = pd.DataFrame({'Models':['LR','DT','RF'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred_pruned)*100,accuracy_score(y_test,y_pred3)*100]})

final_data

sns.barplot(x='Models', y='ACC', data=final_data)



#In the Model RF is Giving Best Accuracy
#There is minor difference between the 3 models



