import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import seaborn as sns

st.title('ML_app')

st.write(""" 
# Explore different classifiers
Which is the best for your purpose?
""")

dataset_name=st.sidebar.selectbox("Select Dataset",("Iris", "Breast Cancer", "Wine Dataset"))
#st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()


    X = data.data
    y = data.target
    return X,y


X,y = get_dataset(dataset_name)
st.write("shape of the datasets",X.shape)
st.write("number of classes", len(np.unique(y)))


def add_parameter(clf_name):
    params = dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    
    elif clf_name=="SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"]=C

    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators

    return params

params=add_parameter(classifier_name)


def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    
    elif clf_name=="SVM":
       clf=SVC(C=params["C"])

    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf


clf = get_classifier(classifier_name,params)


X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)
cr=classification_report(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')
#st.subheader("Confusion Matrix")
#st.write(f'confusion matrix = {cm}')
#st.subheader("Classification Report")
#st.write(f'classification_report = {cr}')

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)


#metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
fig2=plt.figure()
names_pred = np.unique(y_pred)
names_true = np.unique(y_test)
sns.heatmap(cm, square = True,annot = True,fmt='d',cbar=False,xticklabels=names_true,yticklabels=names_pred )
plt.xlabel('True')
plt.ylabel('Predicted')
st.pyplot(fig2)