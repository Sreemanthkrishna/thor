# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:04:23 2019

@author: sreemanth tirumala
GID: G01181388

@author 2: rishikesh pasham
GID: G01188874
"""

import pandas as pd
import numpy as np

# Read csv file into a pandas dataframe.
crime = pd.read_csv('crime.csv',encoding = 'unicode_escape')


crime.isnull().sum() 
#removing columns that are insignificant
for column in crime:
    if(crime[column].count() < 100000):
        crime.drop([column], axis = 1, inplace = True)
crime.head()
crime.isnull().sum()
# Filling the columns with fillna method
crime.fillna({
   'UCR_PART': 'N/A', 
    'DISTRICT': 'N/A', 
    'STREET': 'N/A', 
    'Lat' : 'N/A',
    'Long' : 'N/A'
}, inplace= True)
crime.head()
del crime['REPORTING_AREA']

crime.isnull().sum()
#  Specifying 'INCIDENT_NUMBER' column to use as index.
crime.set_index('INCIDENT_NUMBER', inplace = True)
crime.head()
#renamin columns for easy usage
crime.rename(columns={'INCIDENT_NUMBER': 'Incident_Number', 'OFFENSE_CODE': 'Offense_Code', 'OFFENSE_CODE_GROUP':'Offense_Code_Group', 'OFFENSE_DESCRIPTION': 'Offense_Description', 'DISTRICT': 'District', 'OCCURRED_ON_DATE':'Occured_on_Date', 'YEAR': 'Year', 'MONTH':'Month', 'DAY_OF_WEEK':'Day_of_Week', 'HOUR':'Hour', 'STREET':'Street'}, inplace=True)
crime.head()
# Checking for Duplicate Values
crime.duplicated().sum()
# Removing all the Duplicate Values from the dataset.
crime.drop_duplicates(keep=False, inplace=True)

crime.duplicated().sum()

y = 'Offense_Code_Group'

X = 'DISTRICT','MONTH','DAY_OF_WEEK', 'HOUR','Lat','Long', 'Offense_Code_Group'
crime['Offense_Code_Group'].value_counts().head(15)
crime_model = pd.DataFrame()

i = 0
# Creating a list for offense_code_group

list_offense_code_group = ('Motor Vehicle Accident Response',
                           'Larceny',
                           'Medical Assistance',
                           'Investigate Person',
                           'Other',
                           'Drug Violation',
                           'Simple Assault',
                           'Vandalism',
                           'Verbal Disputes',
                           'Towed',
                           'Investigate Property',
                           'Larceny From Motor Vehicle'
                           'Property Lost'
                           'Warrant Arrests'
                           'Aggravated Assault'
                          )

while i < len(list_offense_code_group):

    crime_model = crime_model.append(crime.loc[crime['Offense_Code_Group'] == list_offense_code_group[i]])
    
    i+=1

list_column = ['District','Month','Day_of_Week',
               'Hour','Lat','Long', 'Offense_Code_Group']


crime_model = crime_model[list_column]
crime_model.head()

# DISTRICT

crime_model['District'] = crime_model['District'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})

crime_model['District'].unique()


crime_model['Month'].unique()
# DAY_OF_WEEK

crime_model['Day_of_Week'] = crime_model['Day_of_Week'].map({
    'Tuesday':2, 
    'Saturday':6, 
    'Monday':1, 
    'Sunday':7, 
    'Thursday':4, 
    'Wednesday':3,
    'Friday':5
})

crime_model['Day_of_Week'].unique()
# HOUR

crime_model['Hour'].unique()

# Lat, Long

crime_model[['Lat', 'Long']].head()
crime_model.fillna(0, inplace = True)
x = crime_model[['District','Month','Day_of_Week','Hour']]
y = crime_model['Offense_Code_Group']
y.unique()
y = y.map({
    'Motor Vehicle Accident Response':1, 
    'Larceny':2, 
    'Medical Assistance':3,
    'Investigate Person':4, 
    'Other':5, 
    'Drug Violation':6, 
    'Simple Assault':7,
    'Vandalism':8, 
    'Verbal Disputes':9, 
    'Towed':10, 
    'Investigate Property':11,
    'Larceny From Motor Vehicle':12
})
    
# Split dataframe into random train and test subsets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    y, 
    test_size = 0.25,
    random_state=42
)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import math
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy import spatial
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
#returns mean ma and min of F1_score 

def fun_results(result):
    print('mean: ' + str(result.mean()))
    print('max: ' + str(result.max()))
    print('min: ' + str(result.min()))

    return 
#Replacing 'N/A' with None (NaN) for reproducing Output
Y_train = Y_train.replace('N/A', None)
X_test = X_test.replace('N/A', None)
Y_test = Y_test.replace('N/A', None)
X_train = X_train.replace('N/A', None)

X_train.isna().sum()
X_test.isna().sum()
Y_train.isna().sum()
Y_test.isna().sum()

Y_test.shape

def fun_DecisionTreeClassifier(X_train, Y_train):
    dec_tree = DecisionTreeClassifier()
    dec_tree = dec_tree.fit(X_train, Y_train)

    dec_tree_pred = dec_tree.predict(X_test)

    dec_tree_score = f1_score(Y_test, dec_tree_pred, average=None)
    accuracy = accuracy_score(Y_test, dec_tree_pred)
    # Determining Accuracy of the model
    print("Accuracy = " + str(math.ceil((accuracy*100))) + str("%"))
    print(confusion_matrix(Y_test, dec_tree_pred))
    print(classification_report(Y_test, dec_tree_pred))
    result = 1 - spatial.distance.cosine(Y_test, dec_tree_pred)
    print(result)
    return fun_results(dec_tree_score)

fun_DecisionTreeClassifier(X_train, Y_train)


def fun_KNeighborsClassifier(X_train, Y_train):
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, Y_train) 
    neigh_pred = neigh.predict(X_test)
    print(neigh_pred.shape)
    neigh_score = f1_score(Y_test, neigh_pred, average=None)
    accuracy = accuracy_score(Y_test, neigh_pred)
    print(confusion_matrix(Y_test, neigh_pred))
    print(classification_report(Y_test, neigh_pred))
    result = 1 - spatial.distance.cosine(Y_test, neigh_pred)
    
    
    # Determining Accuracy of the model.
    print("Accuracy = " + str(math.ceil((accuracy*100))) + str("%"))
    print("cosine similarity")
    print(result)
    return fun_results(neigh_score)

fun_KNeighborsClassifier(X_train, Y_train)

# RandomForestClassifier

def fun_RandomForestClassifier(X_train, Y_train):
    rfc = RandomForestClassifier(n_estimators=10,criterion='gini')
    rfc = rfc.fit(X_train, Y_train)

    rfc_pred = rfc.predict(X_test)

    rfc_score = f1_score(Y_test, rfc_pred, average=None)
    accuracy = accuracy_score(Y_test, rfc_pred)
    print(confusion_matrix(Y_test, rfc_pred))
    print(classification_report(Y_test, rfc_pred))
    print("Accuracy = " + str(math.ceil((accuracy*100))) + str("%"))
    result = 1 - spatial.distance.cosine(Y_test, rfc_pred)
    print("cosine similarity")
    print(result)
    return fun_results(rfc_score)


fun_RandomForestClassifier(X_train, Y_train)

def fun_MNB(X_train, Y_train):
    mnb = MultinomialNB()
    mnb = mnb.fit(X_train, Y_train)

    mnb_pred = mnb.predict(X_test)

    mnb_score = f1_score(Y_test, mnb_pred, average=None)
    accuracy = accuracy_score(Y_test, mnb_pred)
    print(confusion_matrix(Y_test, mnb_pred))
    print(classification_report(Y_test, mnb_pred))
    
    print("Accuracy = " + str(math.ceil((accuracy*100))) + str("%"))
    result = 1 - spatial.distance.cosine(Y_test, mnb_pred)
    print("cosine similarity")
    print(result)
    
    return fun_results(mnb_score)
fun_MNB(X_train, Y_train)

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), 
                                                        X_train, 
                                                        Y_train,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()



#KNN
import matplotlib.pyplot as plt
crime['Lat']= crime['Lat'].replace('N/A', None)
crime['Long']= crime['Long'].replace('N/A', None)

pd.to_numeric(crime['Long'], errors='coerce')


location = crime[['Lat','Long']]


location = location.dropna()
location = location.loc[(location['Lat'] > 40) & (location['Long'] < -60)] 

location.head()
import folium
from folium import plugins

m = folium.Map([42.348624, -71.062492], zoom_start=13)

# plot of Lat and Long cordinates of crimes in Boston
x = location['Long']
y = location['Lat']

colors = np.random.rand(len(x))
print(len(x))


from sklearn.cluster import KMeans
X = location
X = X[~X.isna()]
#K means Clustering #K means  
def doKmeans(X, nclust):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(X, 5)
kmeans = pd.DataFrame(clust_labels)
X.insert((X.shape[1]),'kmeans',kmeans)
#Plot the clusters obtained using k means#Plot the 
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
scatter = ax.scatter(X['Long'],X['Lat'],
                     c=kmeans[0],s=100)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Long')
ax.set_ylabel('Lat')
plt.colorbar(scatter)
plt.show()



