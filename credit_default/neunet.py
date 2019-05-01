import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
##from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix


#import the loan data using pandas
dataset=pd.read_csv("processed_ds.csv")
dataset=dataset.drop('Unnamed: 0',axis=1)
print dataset.head()
dataset1=dataset.drop(['emp_title','earliest_cr_line','mths_since_last_delinq','mths_since_last_major_derog'],axis=1)

X1 = dataset1.iloc[:, 0:8]
X2 = dataset1.iloc[:, 9:]
#X  = pd.concat([X1,X2],axis=1)
y1 = dataset1.iloc[:, 8]
le = preprocessing.LabelEncoder()
le.fit(y1)
##y1=le.transform(y)
##y=pd.DataFrame(data=y1,columns=['loan_status'])

y=data=le.transform(dataset1.iloc[:, 8])

##y=y.apply(le.fit_transform)
#['term', 'sub_grade', 'emp_length', 'home_ownership', 'purpose', 'addr_state', 'open_acc', 'total_acc',  'application_type']
A1=X1.loc[:,'term']
le1 = preprocessing.LabelEncoder()
le1.fit(A1)
A1=pd.DataFrame(data=le1.transform(A1),columns=['term'])
X1=X1.drop('term',axis=1)
X1=pd.concat([A1,X1],axis=1)

A2=X1.loc[:,'sub_grade']
le2 = preprocessing.LabelEncoder()
le2.fit(A2)
A2=pd.DataFrame(data=le2.transform(A2),columns=['sub_grade'])
X1=X1.drop('sub_grade',axis=1)
X1=pd.concat([A2,X1],axis=1)


A3=X1.loc[:,'emp_length']
le3 = preprocessing.LabelEncoder()
le3.fit(A3)
A3=pd.DataFrame(data=le3.transform(A3),columns=['emp_length'])
X1=X1.drop('emp_length',axis=1)
X1=pd.concat([A3,X1],axis=1)

A4=X1.loc[:,'home_ownership']
le4 = preprocessing.LabelEncoder()
le4.fit(A4)
A4=pd.DataFrame(data=le4.transform(A4),columns=['home_ownership'])
X1=X1.drop('home_ownership',axis=1)
X1=pd.concat([A4,X1],axis=1)

A5=X2.loc[:,'purpose']
le5 = preprocessing.LabelEncoder()
le5.fit(A5)
A5=pd.DataFrame(data=le5.transform(A5),columns=['purpose'])
X2=X2.drop('purpose',axis=1)
X2=pd.concat([A5,X2],axis=1)

A6=X2.loc[:,'addr_state']
le6 = preprocessing.LabelEncoder()
le6.fit(A6)
A6=pd.DataFrame(data=le6.transform(A6),columns=['addr_state'])
X2=X2.drop('addr_state',axis=1)
X2=pd.concat([A6,X2],axis=1)

A7=X2.loc[:,'open_acc']
le7 = preprocessing.LabelEncoder()
le7.fit(A7)
A7=pd.DataFrame(data=le7.transform(A7),columns=['open_acc'])
X2=X2.drop('open_acc',axis=1)
X2=pd.concat([A7,X2],axis=1)

A8=X2.loc[:,'total_acc']
le8 = preprocessing.LabelEncoder()
le8.fit(A8)
A8=pd.DataFrame(data=le8.transform(A8),columns=['total_acc'])
X2=X2.drop('total_acc',axis=1)
X2=pd.concat([A8,X2],axis=1)

A9=X2.loc[:,'application_type']
le9 = preprocessing.LabelEncoder()
le9.fit(A9)
A9=pd.DataFrame(data=le9.transform(A9),columns=['application_type'])
X2=X2.drop('application_type',axis=1)
X2=pd.concat([A9,X2],axis=1)

X  = pd.concat([X1,X2],axis=1)


 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

mlp= MLPClassifier(random_state=42,hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
print('Accuracy on the training set: {:.3f}'.format(mlp.score(X_train,y_train)))
print('Accuracy on the test set: {:.3f}'.format(mlp.score(X_test,y_test)))
predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
##pipeline = make_pipeline(StandardScaler(), MLPRegressor(solver='lbfgs', hidden_layer_sizes=50))
##pipeline.fit(X_train, y_train)                                                                  
##pipeline.score(X_test, y_test)
