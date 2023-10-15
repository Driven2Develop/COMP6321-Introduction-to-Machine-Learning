import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.mixture
import sklearn.model_selection
import sklearn.metrics
import sklearn.utils
import sklearn.preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

path = './creditcard.csv'
data = pd.read_csv(path)
X = data.iloc[:, 1: data.shape[1] - 1]
y = data.iloc[:, -1]

X = sklearn.preprocessing.StandardScaler().fit_transform(X)

c=np.logspace(start=0,stop=3,num=4)
g=np.logspace(start=-2,stop=3,num=4)

hyperprameters={
    'C':c,
    'gamma':g
}

model=SVC(random_state=0)
skf=StratifiedKFold(n_splits=2)
cross_validation=GridSearchCV(model,param_grid=hyperprameters,verbose=1,cv=skf)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.8,random_state=0)

cv_model=cross_validation.fit(X_train,y_train)