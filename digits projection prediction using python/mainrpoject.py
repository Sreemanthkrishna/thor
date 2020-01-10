

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.externals import joblib

import pickle as pkl



data = pd.read_csv("pen.csv") 
print("Dataset Lenght:",len(data))
print("Dataset Dimensions:",data.shape)
infeatures=data.iloc[:,0:16]
print(infeatures)
print(" Shape:",infeatures.shape)
outfeature=data.iloc[:,-1]
print (outfeature)
print("out Shape:",outfeature.shape)
totest=pd.read_csv("pentest.csv")
totestin=totest.iloc[:,0:16]
totestout=totest.iloc[:,-1]

# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(infeatures, outfeature, test_size=0.25)
# random forest model creation
rfc = RandomForestClassifier(n_estimators=10,criterion='gini')
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))

#kf = KFold(n_splits=2) # Define the split - into 2 folds 
#kf.get_n_splits(infeatures) # returns the number of splitting iterations in
#print(kf) 
#KFold(n_splits=2, random_state=None, shuffle=False)

scores=cross_val_score(rfc, infeatures, outfeature, cv=5)
print(scores)

rfcmodel = 'rfcmodel.pkl'
# Open the file to save as pkl file
rfcmodel = open(rfcmodel, 'wb')
pkl.dump(rfc, rfcmodel)
# Close the pickle instances
rfcmodel.close()

rfcfrompkl = joblib.load("rfcmodel.pkl")

# implementing train-test-split
X_traintotest, X_testtotest, y_traintotest, y_testtotest = train_test_split(totestin, totestout, test_size=0.25)
rfc_predtest=rfcfrompkl.predict(X_testtotest)
print("Accuracy:",metrics.accuracy_score(y_testtotest,rfc_predtest))

  



