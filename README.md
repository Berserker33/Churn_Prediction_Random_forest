# Churn_Prediction_Random_forest

import numpy as np 
import pandas as pd 

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Importing
telcom = pd.read_csv(r"C:\Users\Shiva\Downloads\Random Forest\churn.csv")
telcom.head()

# 2. Data Pre-Processing 
# 2.1 Removing Irrelevant Variable
telcom = telcom.drop('customerID', axis=1)

# 2.2 Data Type and Conversion
telcom.info()

# Replace spaces with NaN and convert TotalCharges to float
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", np.nan).astype(float)

# Fill missing values with the mean
telcom.TotalCharges.fillna(telcom.TotalCharges.mean(), inplace=True)

# 3. Missing Value Identification & Treatment
telcom.isnull().sum()

# 4. Outlier Identification & Treatment
sns.boxplot(y=telcom.TotalCharges)
sns.boxplot(y=telcom.MonthlyCharges)
sns.boxplot(y=telcom.tenure)

# 5. Data Manipulation
# Replace binary numeric values with Yes/No
telcom.SeniorCitizen = telcom.SeniorCitizen.replace({1: "Yes", 0: "No"})

# Replace "No internet service" and "No phone service" with "No"
columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in columns_to_replace:
    telcom[col] = telcom[col].replace({'No internet service': 'No'})
telcom.MultipleLines = telcom.MultipleLines.replace({'No phone service': 'No'})

# Create tenure groups
def tenure_lab(telcom):
    if telcom["tenure"] <= 12:
        return "Tenure_0-12"
    elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24):
        return "Tenure_13-24"
    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
        return "Tenure_25-48"
    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60):
        return "Tenure_49-60"
    elif telcom["tenure"] > 60:
        return "Tenure_gt_60"

telcom["tenure_group"] = telcom.apply(lambda x: tenure_lab(x), axis=1)

# 6. Data Visualization
import plotly.express as px

fig = px.pie(telcom, names='Churn', color='Churn',
             color_discrete_map={'Yes': 'red',
                                 'No': 'green'})
fig.show()

# 7. Label Encoding
telcom_num = telcom.select_dtypes(include=[np.number])
telcom_dummies = telcom.select_dtypes(include=['object'])

from sklearn.preprocessing import LabelEncoder
telcom_dummies = telcom_dummies.apply(LabelEncoder().fit_transform)

# Combine numerical and encoded categorical data
telcom = pd.concat([telcom_num, telcom_dummies], axis=1)

# 8. Data Partition
from sklearn.model_selection import train_test_split

X = telcom.drop('Churn', axis=1)
Y = telcom[['Churn']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1234)

# 9. Model -- Random Forest
from sklearn.ensemble import RandomForestClassifier

RFModel = RandomForestClassifier(random_state=20,
                                  n_estimators=25,
                                  criterion="gini",
                                  max_depth=4,
                                  min_samples_split=100,
                                  min_samples_leaf=50,
                                  max_features="sqrt")

RFModel.fit(X_train, y_train)

# Feature Importance
imp = pd.Series(data=RFModel.feature_importances_, index=RFModel.feature_names_in_).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
plt.title("Feature Importance / Selection")
sns.barplot(y=imp.head().index, x=imp.head().values, palette="BrBG")

# 10. Random Forest Visualization
from sklearn.tree import export_graphviz
import pydot

tree = RFModel.estimators_[4]

export_graphviz(tree, out_file='abc.dot', 
                feature_names=list(X.columns),
                class_names=['No', 'Yes'],
                rounded=True, 
                filled=True)
(graph,) = pydot.graph_from_dot_file('abc.dot')
graph.write_png('tree.png')

from IPython.display import Image
Image(filename='tree.png')

# 11. Predictions on Train Dataset
train = pd.concat([X_train, y_train], axis=1)
train['Predicted'] = RFModel.predict(X_train)

# Model Performance Metrics on Train Data
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix_train = confusion_matrix(train['Predicted'], train['Churn'])
print(conf_matrix_train)

train_accuracy = ((conf_matrix_train[0, 0] + conf_matrix_train[1, 1]) / len(train)) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

print(classification_report(train['Churn'], train['Predicted']))

# 12. Predictions on Test Dataset
test = pd.concat([X_test, y_test], axis=1)
test['Predicted'] = RFModel.predict(X_test)

# Model Performance Metrics on Test Data
conf_matrix_test = confusion_matrix(test['Predicted'], test['Churn'])
print(conf_matrix_test)

test_accuracy = ((conf_matrix_test[0, 0] + conf_matrix_test[1, 1]) / len(test)) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

print(classification_report(test['Churn'], test['Predicted']))
