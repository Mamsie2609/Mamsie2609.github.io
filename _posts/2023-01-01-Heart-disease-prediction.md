---
layout: post
title: Heart Disease Prediction
image: "/posts/heart_disease.jpg"
tags: [Heart Disease, Machine Learning, Python]
---

This project aims to build an ML model that can predict the likelihood of a person having a heart disease based on the given features.

---

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Exploratory Data Analysis](#exploratory-data-analysis)
- [04. Univariate Analysis](#univariate-analysis)
- [05. Bivariate Analysis](#bivariate-analysis)
- [06. Multivariate Analysis](#multiivariate-analysis)
- [07. Feature Engineering & Data Preprocessing](#data-preprocessing) 
- [08. Machine Learning](#machine-learning)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Peterside Hospital embarked on a significant project to enhance its patient care and diagnostic capabilities. To achieve this, they engaged the services of a skilled and experienced data scientist to leverage the power of machine learning and predictive analytics. The primary objective of the project was to develop a robust model capable of accurately predicting the likelihood of heart disease in patients based on a range of key features, including age, sex, and chest pain type.
<br>
<br>
### Actions <a name="overview-actions"></a>

To commence the heart disease prediction project, the first step was to compile the essential data by collecting key patient metrics that could potentially aid in predicting the likelihood of heart disease. This involved gathering relevant information from various sources and databases.

Next, I proceeded to test eight different machine learning models as part of the predictive modelling process. These models included:

* Logistic Regression
* Random Forest
* XGBoostClassifier
* K Nearest Neighbours (KNN)
* Stochastic Gradient Descent Classifier (SGD)
* Support Vector Classifier (SVC)
* Gaussian Naive Bayes
* Decision Tree

### Results <a name="overview-results"></a>

Based on the performance metrics, the SGD Classifier model outperformed the other models in terms of accuracy, precision, recall, and ROC score. 


<br>

**Metric 1: Accuracy Score**

* Logistic Regression = 86.89%
* Random Forest = 83.61%
* XGBoostClassifier = 78.69%
* KNN = 78.69%
* SGD = 88.52%
* SVC = 65.57%
* Gaussian Naive Bayes = 86.89%
* Decision Tree = 78.69%
  
<br>

**Metric 2: Precision Score**

* Logistic Regression = 87.5%
* Random Forest = 84.38%
* XGBoostClassifier = 82.76%
* KNN = 78.79%
* SGD = 87.88%
* SVC = 65.71%
* Gaussian Naive Bayes = 90.0%
* Decision Tree = 82.76%

<br>

**Metric 3: Recall**

* Logistic Regression = 86.89%
* Random Forest = 84.38%
* XGBoostClassifier = 75%
* KNN = 81.25%
* SGD = 90.62%
* SVC = 71.88%
* Gaussian Naive Bayes = 84.38%
* Decision Tree = 75%

<br>

**Metric 4: ROC Score**

* Logistic Regression = 86.89%
* Random Forest = 83.57%
* XGBoostClassifier = 78.88%
* KNN = 78.56%
* SGD = 88.42%
* SVC = 65.25%
* Gaussian Naive Bayes = 87.02%
* Decision Tree = 78.88%
<br>
<br>

# Data Overview  <a name="data-overview"></a>

| **Variable Name** | **Description** |
|---|---|
|age | age in years|
|sex |1 = male; 0 = female|
|cp | chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4; asymptomatic)|
|trestbos | resting blood pressure (in mm Hg on admission to the hospital)|
|chol | serum cholesterol in mg/dl|
|fbs |fasting blood sugar > 120 mg/dl (1 = true; 0 = false)|
|restecg | resting electrocardiographic results|
|thalach | maximum heart rate achieved|
|exang | exercise-induced angina (1 = yes; 0 = no)|
|oldpeak | ST depression induced by exercise relative to rest|
|slope | the slope of the peak exercise ST segment|
|ca | number of major vessels (0-3) colored by fluorosopy|
|thal | 3 = normal; 6 = fixed defect: 7 = reversible defect|
|target | have disease or not (1=yes, 0=no)|

<br>

# Modelling Overview  <a name="modelling-overview"></a>

I will build a model that looks to accurately predict *heart_disease*, based on the patients' metrics listed above. I tested eight different machine learning models. These models included:

* Logistic Regression
* Random Forest
* XGBoostClassifier
* K Nearest Neighbours (KNN)
* Stochastic Gradient Descent Classifier (SGD)
* Support Vector Classifier (SVC)
* Gaussian Naive Bayes
* Decision Tree

```python

# import necessary libraries 
# for data analysis
import pandas as pd
import numpy as np

# for data visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

# for data pre-processing.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# classifier libraries
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.svm import LinearSVC, SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

# evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
from sklearn.metrics import confusion_matrix

import warnings 
warnings.filterwarnings ("ignore")
```
<br>

##### Data Verification - data type, number of features and rows, missing data, etc 


```python

df.info()

```

##### Statistical analysis of the data

```python

df.describe()

```

##### Check for missing values

```python

print(df.isnull().sum())

```

##### Visualising the missing data 

```python

plt.figure(figsize = (10,3))
sns.heatmap(df.isnull(), cbar=True, cmap="Blues_r");

```

<br>
![alt text](/img/posts/missing_data.png "missing data")

<br>

# Exploratory data analysis <a name="exploratory-data-analysis"></a>

<br>

## Univariate analysis <a name="univariate-analysis"></a>

<br>

##### Return the column labels (Column Names) of the data frame 

```python

df.columns

```
<br>

##### Check for outliers 

```python

sns.boxplot(x=df["thalassemia"]);

```

<br>
![alt text](/img/posts/thalassemia.png "thalassemia")

<br>

##### Check for outliers 

```python

sns.boxplot(x=df["cholesterol"]);

```

<br>
![alt text](/img/posts/cholesterol.png "cholesterol")

<br>

##### Check for outliers 

```python

sns.boxplot(x=df["resting_blood_pressure"]);

```

<br>
![alt text](/img/posts/resting_blood_pressure.png "resting_blood_pressure")

<br>

##### Check for outliers 

```python

sns.boxplot(x=df["max_heart_rate_achieved"]);

```

<br>
![alt text](/img/posts/max_heart_rate_achieved.png "max_heart_rate_achieved")

<br>

* The plots above indicate the presence of outliers in several features of the dataset.


<br>

```python

# Data Visualisation
# Age Bracket
def age_bracket(age): 
    if age <= 35:
        return "Young adults"
    elif age <= 55:
        return "Middle-aged adults" 
    elif age <= 65:
        return "Senior citizens" 
    else:
        return "Elderly"
df['age_bracket'] = df['age'].apply(age_bracket)

# Investigating the age group of patients

# Sets the size of the plot to be 10 inches in width and 5 inches in height.
plt.figure(figsize = (10, 5))

# Creates a countplot using the Seaborn library, with 'age_bracket' on the x-axis 
sns.countplot (x='age_bracket', data=df) 

# Sets the label for the x-axis as 'Age Group'.
plt.xlabel('Age Group')

# Sets the label for the y-axis as 'Count of Age Group'.
plt.ylabel('Count of Age Group') 

# Sets the title of the plot as 'Total Number of Patients'.
plt.title('Total Number of Patients');

```

<br>
![alt text](/img/posts/Total_Number_of_Patients.png "Total Number of Patients")

* Among the patients in the hospital, the age groups with the highest number of individuals are middle-aged and older adults, typically between the ages of 36 to 65. Conversely, young adults make up the smallest proportion of patients.

<br>

```python

# Data Visualisation

# Investigate the gender distribution of the patients. 
def Gender (sex) :
    if sex == 1:
        return "Male"
    else:
        return "Female"
    
df['Gender'] = df['sex'].apply(Gender)

# Convert the gender counts into a dataframe
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

# Create a pie chart
plt.figure(figsize=(10, 5))
plt.pie(gender_counts['Count'], labels=gender_counts['Gender'], startangle=90, counterclock=False, autopct='%1.1f%%')
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Add a circle at the center to create a doughnut chart
plt.gca().axis('equal')
plt.title('Gender Distribution')
plt.tight_layout()
plt.show()

```

<br>
![alt text](/img/posts/Gender_distribution.png "Gender distribution")

<br>

* The hospital has twice as many male patients as female patients, with males accounting for 68.3% of the total patients and females comprising 31.7%.


<br>

```python

# Data Visualisation

# Visualise the distribution of patients based on the different categories of chest pain.

def chest_pain(cp):
    if cp == 1:
        return "Typical Angina"
    elif cp == 2:
        return "Atypical Angina" 
    elif cp == 3:
        return "Non-anginal Pain"
    else:
        return "Asymptomatic"
    
df['cp_cat'] = df['chest_pain_type'].apply (chest_pain)

plt. figure(figsize = (10, 5))
sns.countplot(x='cp_cat', data=df)  
plt.xlabel ("Types of chest pain") 
plt.ylabel("Count of patient Gender") 
plt. title("Total Number of Patients");

```

<br>
![alt text](/img/posts/distribution of patients based on the different categories of chest pain.png "Total Number of Patients based on the different categories of chest pain")

<br>

* Regarding chest pain categories, patients with the highest number are asymptomatic, followed by those with atypical angina, typical angina, and finally, those with non-anginal pain, who are the least represented.


<br>

```python

# Data Visualisation

# target - have disease or not (1=yes, 0=no)
def label(tg):
    if tg == 1:
        return "Yes" 
    else:
        return "No"
df['label'] = df['target'].apply(label)

# Total patient in each category
print(df["label"].value_counts ())

# Investigate the gender distribution of patients in each category of 'label'.
plt.figure(figsize = (10, 5))
sns.countplot (x='label', data=df) 
plt.xlabel('Target')
plt.ylabel ('Count of patient Gender')
plt. title( 'Total Number of Patients');

```

<br>
![alt text](/img/posts/gender distribution of patients in each category of 'label'.png "Gender distribution based on the categories of target")

<br>

* There are slightly more patients who have been diagnosed with heart disease than those who have not.

<br>

# Bivariate Analysis<a name="bivariate-analysis"></a>

##### Visualising the distribution of the age group of patients and whether they have a disease (represented by the 'label' column)

```python

plt.figure(figsize = (10, 5))
sns.countplot(x='age_bracket', data=df, hue='label') 
plt.xlabel('Age Group')
plt.ylabel ('Count of Age Group')
plt.title('Total Number of Patients');

```

<br>
![alt text](/img/posts/distribution of the age group of patients whether they have a disease or not.png "Age distribution based on the categories of target")

<br>

* In terms of age groups, middle-aged adults have a higher incidence of diagnosed heart disease compared to those without a diagnosis. Among senior citizens, the proportion of individuals without a heart disease diagnosis is greater than those with a diagnosis. In the elderly population, the number of patients with a heart disease diagnosis is slightly higher than those without. For young adults, there are more individuals with a heart disease diagnosis than without.

<br>

## Visualising the distribution of gender of patients whether they have a disease (represented by the 'label' column)

```python
plt.figure(figsize = (10, 5))
sns.countplot(x='Gender', data=df, hue='label') 
plt.xlabel('Gender')
plt.ylabel ('Count of patient Gender')
plt.title('Total Number of Patients');

```

<br>
![alt text](/img/posts/distribution of gender of patients whether they have a disease or not.png "Gender distribution based on the categories of target")

<br>

* The incidence of diagnosed heart disease is relatively higher in males than in females. Additionally, the number of males without a heart disease diagnosis is higher than that of females without a diagnosis.

<br>

##### Shows the distribution of the 'label' among different categories of chest pain ('cp_cat').

```python

plt.figure(figsize = (10, 5))
sns.countplot(x='cp_cat', data=df, hue='label') 
plt.xlabel('Types of chest pain')
plt.ylabel ('Count of patient Gender')
plt.title('Total Number of Patients');

```

<br>
![alt text](/img/posts/distribution of the 'label' among different categories of chest pain.png "Label distribution based on the categories of chest pain")

<br>

# Multivariate Analysis <a name="multiivariate-analysis"></a>

##### Correlation between the features in the dataset

```python

plt.figure(figsize = (10, 10))
hm = sns.heatmap(df.corr(), cbar=True, annot=True, square=True, fmt='.2f',
annot_kws={'size': 10})

```

<br>
![alt text](/img/posts/heatmap.png "Correlation between the features in the dataset")

<br>

* In general, the dataset features show weak correlations. However, a moderate positive correlation of 0.43 is observed between chest pain type and target. On the other hand, the most pronounced negative correlation of -0.58 is found between st slope and st depression.

<br>

# Feature Engineering & Data Preprocessing <a name="data-preprocessing"></a>

##### Create a copy of the data (Exclude target/Label alongside other columns that were created)

```python

df1 = df[['age','chest_pain_type','resting_blood_pressure', 'cholesterol' , 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression','st_slope', 'num_major_vessels','thalassemia']]
label = df [ ['target' ]]

```

<br>

## Dealing with Outliers - 'resting_blood pressure', 'cholesterol', 'thalassemia'

<br>

##### Normalize the data

```python

scaler = MinMaxScaler()

df1 ["Scaled RBP"] = scaler.fit_transform(df1['resting_blood_pressure']. values.reshape(-1, 1))
df1 ["Scaled chol"] = scaler.fit_transform(df1 [ ['cholesterol']].values.reshape (-1, 1))
df1 ["Scaled_thal"] = scaler.fit_transform(df1 [[ 'thalassemia']].values.reshape (-1, 1))
df1 ["Scaled_max_heart_rate"] = scaler.fit_transform(df1[[ 'max_heart_rate_achieved']].values.reshape(-1, 1))

```
                                              
##### Removes the columns 'resting_blood_pressure', 'thalassemia', 'cholesterol', 'max_heart_rate_achieved' from the DataFrame 'df1' and updates the DataFrame in place.

```python

df1.drop(['resting_blood_pressure', 'thalassemia', 'cholesterol', 'max_heart_rate_achieved'], axis=1, inplace=True)

```

<br>

# Machine Learning

##### Split the dataset into training and testing sets 


```python

X_train, X_test, y_train, y_test = train_test_split(df1, label, test_size=0.2, random_state=42)

```
<br>

### Model Building
#### Logistic Regression

```python

logreg = LogisticRegression()

logreg.fit (X_train, y_train)

ly_pred = logreg.predict (X_test)

print("Logistic Regression")
print ("Accuracy:", accuracy_score (y_test, ly_pred))
print("Precision:", precision_score(y_test, ly_pred))
print("Recall:", recall_score (y_test, ly_pred))
print("F1-score:" , f1_score (y_test, ly_pred))
print("AUC-ROC:", roc_auc_score(y_test, ly_pred))

#### Logistic Regression
* Accuracy: 0.8688524590163934
* Precision: 0.875
* Recall: 0.875
* F1-score: 0.875
* AUC-ROC: 0.8685344827586206

```

* The logistic regression model achieved an accuracy of 0.87, indicating that the model correctly predicted the presence or absence of heart disease in 87% of cases. The precision score of 0.88 indicates that out of all the positive predictions made by the model, 88% were actually true positives. The recall score of 0.88 indicates that the model was able to correctly identify 88% of all positive cases of heart disease. The F1-score of 0.88 indicates a good balance between precision and recall. The AUC-ROC score of 0.87 indicates that the model is good at distinguishing between positive and negative cases of heart disease.


##### Create a confusion matrix

```python

lcm = confusion_matrix(y_test, ly_pred)

```

##### Visualise the confusion matrix


```python

sns.heatmap(lcm, annot=True, cmap="Blues", fmt="g") 
plt.xlabel("Predicted") 
plt.ylabel ("Actual") 
plt.title("Confusion Matrix") 
plt.show()

```

<br>
![alt text](/img/posts/Confusion_Matrix.png "Confusion matrix")

* True Positive (TP): The model correctly predicted 25 individuals as having a heart disease.
* False Positive (FP): The model incorrectly predicted 4 individuals as having a heart disease when they actually did not.
* False Negative (FN): The model incorrectly predicted 4 individuals as not having a heart disease when they actually did.
* True Negative (TN): The model correctly predicted 28 individuals as not having a heart disease.
* Overall, the model appears to perform reasonably well in identifying positive cases.

<br>

#### Random Forest

#### Model Building
#### Random Forest Classifier

```python

rfc = RandomForestClassifier ()
rfc.fit(X_train, y_train) 
rfy_pred = rfc.predict (X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, rfy_pred))
print("Precision:", precision_score(y_test, rfy_pred))
print("Recall:", recall_score (y_test, rfy_pred))
print("F1-score:", f1_score (y_test, rfy_pred))
print("AUC-ROC:", roc_auc_score(y_test, rfy_pred))

```

#### Random Forest
* Accuracy: 0.8360655737704918
* Precision: 0.84375
* Recall: 0.84375
* F1-score: 0.84375
* AUC-ROC: 0.8356681034482758

<br>

##### Create a confusion matrix


```python

rcm = confusion_matrix(y_test, rfy_pred)

```

##### Visualise the confusion matrix

```python

sns.heatmap(rcm, annot=True, cmap="Blues", fmt="g") 
plt.xlabel("Predicted") 
plt.ylabel ("Actual") 
plt.title("Confusion Matrix") 
plt.show()

```

<br>
![alt text](/img/posts/Confusion_Matrix2.png "Confusion matrix")

* The confusion matrix shows that out of the total 61 test cases, 24 were true positive and 26 were true negative. The model predicted 5 cases as positive which were actually negative (false positive), and 6 cases as negative which were actually positive (false negative).

<br>

#### 8 Machine learning Algorithms will be applied to the dataset

```python

classifiers = [[XGBClassifier(), 'XGB Classifier'],
              [RandomForestClassifier(), 'Random Forest'],
              [KNeighborsClassifier(),'K-Nearest Neighbors'],
              [SGDClassifier(), 'SGD Classifier'],
              [SVC(), 'SVC'],
              [GaussianNB(),'Naive Bayes'],
              [DecisionTreeClassifier(random_state = 42), "Decision tree"],
              [LogisticRegression(),'Logistic Regression']
              ]

```

<br>

```python

acc_list = {}
precision_list = {}
recall_list = {}
roc_list = {}
cm_dict = {}
f1_list = {}

for classifier in classifiers:
    model = classifier[0]
    model.fit(X_train, y_train) 
    model_name = classifier[1]
    
    pred = model.predict(X_test)
    
    a_score = accuracy_score (y_test, pred)
    p_score = precision_score(y_test, pred)
    r_score = recall_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    acc_list[model_name] = ([str(round (a_score*100, 2)) + '%'])
    precision_list[model_name] = ([str(round(p_score*100, 2)) + '%'])
    recall_list[model_name] = ([str(round(r_score*100, 2)) + '%'])
    roc_list[model_name] = ([str(round(roc_score*100, 2)) + '%'])
    f1_list[model_name] = [str(round(f1*100, 2)) + '%']  
    
    cm = confusion_matrix(y_test, pred)
    cm_dict[model_name] = cm

    if model_name != classifiers[-1][1]:
       print('')        
```

<br>

**Metric 1: Accuracy Score**

* Logistic Regression = 86.89%
* Random Forest = 83.61%
* XGBoostClassifier = 78.69%
* KNN = 78.69%
* SGD = 88.52%
* SVC = 65.57%
* Gaussian Naive Bayes = 86.89%
* Decision Tree = 78.69%
  
<br>

**Metric 2: Precision Score**

* Logistic Regression = 87.5%
* Random Forest = 84.38%
* XGBoostClassifier = 82.76%
* KNN = 78.79%
* SGD = 87.88%
* SVC = 65.71%
* Gaussian Naive Bayes = 90.0%
* Decision Tree = 82.76%

<br>

**Metric 3: Recall**

* Logistic Regression = 86.89%
* Random Forest = 84.38%
* XGBoostClassifier = 75%
* KNN = 81.25%
* SGD = 90.62%
* SVC = 71.88%
* Gaussian Naive Bayes = 84.38%
* Decision Tree = 75%

<br>

**Metric 4: ROC Score**

* Logistic Regression = 86.89%
* Random Forest = 83.57%
* XGBoostClassifier = 78.88%
* KNN = 78.56%
* SGD = 88.42%
* SVC = 65.25%
* Gaussian Naive Bayes = 87.02%
* Decision Tree = 78.88%
<br>
<br>

* Based on the performance metrics, the SGD Classifier model outperformed the other models in terms of accuracy, precision, recall, and ROC score. Consequently, it can be concluded that the SGD Classifier model would be the most suitable option for predicting the probability of an individual having heart disease.
