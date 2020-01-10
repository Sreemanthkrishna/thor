

from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.externals import joblib
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE 
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix







rfcfrompkl = joblib.load("rfcmodel.pkl")
data = pd.read_csv("imp.csv") 
print("Dataset Lenght:",len(data))
print("Dataset Dimensions:",data.shape)
infeatures=data.iloc[:,1:784]
outfeatures=data.iloc[:,0]
scaler = StandardScaler()
scaler.fit(infeatures)

pca = PCA(n_components=16)
principalComponents = pca.fit_transform(infeatures)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1','principal component 2','principal component 3','principal component 4','principal component 5','principal component 6','principal component 7','principal component 8','principal component 9','principal component 10','principal component 11','principal component 12','principal component 13','principal component 14','principal component 15','principal component 16'])

principalDf=principalDf.astype(int)


X_train, X_test, y_train, y_test = train_test_split(principalDf, outfeatures, test_size=0.25)
rfc_predict = rfcfrompkl.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))


scaler1 = MinMaxScaler()
scaler1.fit(principalDf)


principalDf1=scaler1.transform(principalDf)

principalDf1=principalDf1.astype(int)
print(principalDf1)

X_train, X_test, y_train, y_test = train_test_split(principalDf1, outfeatures, test_size=0.30)
rfc_predict = rfcfrompkl.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, rfc_predict))

print(f1_score(y_test, rfc_predict, average="macro"))
print(precision_score(y_test, rfc_predict, average="macro"))
print(recall_score(y_test, rfc_predict, average="macro")) 




