### Step 1: Loading the Data
df = pd.read_csv('/workspaces/KaggleProject/train.csv')


### Step 2: Handling Missing Values
missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0].index
df[missing_columns] = df[missing_columns].fillna(df[missing_columns].mean())
df = df.dropna()


### Step 3: Correcting Data Types
  
df['date_column'] = pd.to_datetime(df['date_column'])
df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')



### Step 4: Handling Outliers

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

df.to_csv('cleaned_data.csv', index=False)
