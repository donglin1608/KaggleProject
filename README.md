# KaggleProject 
### THIS IS STANDARD PROCESS TO CLEAN DATA BY USING PYTHON, THIS DOCUMENT IS AIMING HELP ANYONE TO CLEAN DATA WITH PYTHON. 
### HOWEVER, IT IS BEING USEFUL FOR ME TO FOLLOW THIS PROCESS, INSURE THAT I WON'T MISS ANYTHING THAT NEEDS TO ADDRESS. 

## Overview 
This is project involves cleaning and preparing a dataset from Kaggle for further analysis. 
The goal is to clean the data by addressing missing values, correcting data types, handling outliers, 
and normalizing the data to ensure it's ready for analysis.


## Assumptions
- The dataset is in CSV format.
- All missing values are treated as missing data points that need to be addressed.
- Categorical variables need to be encoded for future use in machine learning models.
- Outliers is identified and handled to avoid skewing the results.


## Data Cleaning Process

### Step 1: Loading the Data
- Import the dataset using `pandas`:
  ```python
  import pandas as pd
  df = pd.read_csv('data.csv')


### Step 2: Handling Missing Values
- Import the dataset using `pandas`:
  ```python
  import pandas as pd
  df = pd.read_csv('data.csv')

- Drop off missing values

### Step 3: Correcting Data Types
  ```python
 df['date_column'] = pd.to_datetime(df['date_column'])
df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')
```


### Step 4: Handling Outliers
standard formula: 
Q1 = df['numeric_column'].quantile(0.25)
Q3 = df['numeric_column'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['numeric_column'] < (Q1 - 1.5 * IQR)) | (df['numeric_column'] > (Q3 + 1.5 * IQR))]
df = df[~df.index.isin(outliers.index)]

### Step 5: Normalizing Data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['normalized_column'] = scaler.fit_transform(df[['numeric_column']])

### Step 6: Encoding Categorical Variables

df = pd.get_dummies(df, columns=['categorical_column'])

### Step 7: Saving the Cleaned Data

