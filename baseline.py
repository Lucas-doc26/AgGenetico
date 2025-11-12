import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import time

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

y_train = train['label'].values 
x_train = train.drop('label', axis=1)

y_test = test['label'].values 
x_test = test.drop('label', axis=1)

clf = tree.DecisionTreeClassifier(random_state=1)

inicio = time.time()
clf = clf.fit(x_train, y_train)
fim = time.time()
print("Tempo de treinamento:", fim - inicio, "segundos")

y_pred = clf.predict(x_test)

print("Acurácia:", accuracy_score(y_test, y_pred))