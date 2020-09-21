#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

#read the datset

df= pd.read_csv("C:/Users/Harsh Anand/Desktop/class/interview preparation/Email Performance Analytics _Data.csv", sep='\t')


df = df[df['Eml.Sent.Num'].notna()]

#label encoder:-
# I = 0
# o = 1
# U = 2
lb_make = LabelEncoder()
df["Emailpermissionstatus"] = lb_make.fit_transform(df["Emailpermissionstatus"])

# save the target variable and delete it with the main file

y = df['Emailpermissionstatus']
del df['Emailpermissionstatus']

# Droped unwanted features
df.drop(['Midascontactid', 'Createdat','Email.Pref.Ch.Dt','Cont.Acq.Dte','Prod.Buy.Last','Prod.Buy.First','Country','Eml.Sent.First.Ts','Eml.Sent.Last.Ts','Eml.Open.First.Ts','Eml.Open.Last.Ts','Eml.Click.First.Ts','Eml.Click.Last.Ts','Email.Domain','Em.Pref.Status','Em.Spamclplt.Dte','Em.Pref.Dte','Segmentsecondary','Entry.Point.List'], axis = 1,inplace=True)



df['Email.Acq.Dt'] = df.apply(lambda row : 0 if pd.isnull(row["Email.Acq.Dt"]) else 1 ,axis=1)
df['Prod.Buy.Accessories'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Accessories"]) else 1 ,axis=1)
df['Prod.Buy.All.In.One.Desktop'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.All.In.One.Desktop"]) else 1 ,axis=1)
df['Prod.Buy.Desktop'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Desktop"]) else 1 ,axis=1)
df['Prod.Buy.Idc'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Idc"]) else 1 ,axis=1)
df['Prod.Buy.Idpd'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Idpd"]) else 1 ,axis=1)
df['Prod.Buy.Laptop'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Laptop"]) else 1 ,axis=1)
df['Prod.Buy.Servers'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Servers"]) else 1 ,axis=1)
df['Prod.Buy.Software'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Software"]) else 1 ,axis=1)
df['Prod.Buy.Thnkc'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Thnkc"]) else 1 ,axis=1)
df['Prod.Buy.Thnkp'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Thnkp"]) else 1 ,axis=1)
df['Prod.Buy.Warranties'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Warranties"]) else 1 ,axis=1)
df['Prod.Buy.Workstation'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Workstation"]) else 1 ,axis=1)
df['Prod.Buy.Android.Tablet'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Android.Tablet"]) else 1 ,axis=1)
df['Prod.Buy.Windows.Tablet'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Windows.Tablet"]) else 1 ,axis=1)
df['Prod.Buy.Convertible'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Convertible"]) else 1 ,axis=1)
df['Prod.Buy.Android.Os'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Android.Os"]) else 1 ,axis=1)
df['Prod.Buy.Gaming'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Gaming"]) else 1 ,axis=1)
df['Prod.Buy.Windows.Os'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Windows.Os"]) else 1 ,axis=1)
df['Prod.Buy.Gaming.Related'] = df.apply(lambda row : 0 if pd.isnull(row["Prod.Buy.Gaming.Related"]) else 1 ,axis=1)
df['Eml.Bounce.Last.Ts'] = df.apply(lambda row : 0 if pd.isnull(row["Eml.Bounce.Last.Ts"]) else 1 ,axis=1)
df['Eml.Unsub.Last.Ts'] = df.apply(lambda row : 0 if pd.isnull(row["Eml.Unsub.Last.Ts"]) else 1 ,axis=1)
df['Eml.Spam.Last.Ts'] = df.apply(lambda row : 0 if pd.isnull(row["Eml.Spam.Last.Ts"]) else 1 ,axis=1)
df['Invalid.Email.Dte'] = df.apply(lambda row : 0 if pd.isnull(row["Invalid.Email.Dte"]) else 1 ,axis=1)


# use one_hot_encoding for this 

lb_style = LabelBinarizer()
cont_acq_src = lb_style.fit_transform(df["Cont.Acq.Src"].astype(str))
df1 = pd.DataFrame(cont_acq_src, columns=lb_style.classes_)
df1.drop(['nan'], axis = 1,inplace=True)


email_src_first = lb_style.fit_transform(df["Email.Src.First"].astype(str))
df2 = pd.DataFrame(email_src_first, columns=lb_style.classes_)
df2.drop(['nan'], axis = 1,inplace=True)



segment = lb_style.fit_transform(df["Segment"].astype(str))
df3 = pd.DataFrame(segment, columns=lb_style.classes_)

cont_acq_src2 = lb_style.fit_transform(df["Cont.Acq.Src2"].astype(str))
df4 = pd.DataFrame(cont_acq_src2, columns=lb_style.classes_)
df4.drop(['nan'], axis = 1,inplace=True)


df= df.reset_index(drop=True)
df = pd.concat([df,df1,df2,df3,df4], axis=1)



df.drop(["Cont.Acq.Src","Email.Src.First","Segment","Cont.Acq.Src2"], axis = 1,inplace=True)




X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify = y, random_state = 1)


X_train_s = X_train
X_test_s = X_test




model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train_s, y_train)



y_train_pred = model.predict(X_train_s)
y_test_pred = model.predict(X_test_s)


# Training Metrice by the help of confusion matrix

print("Training Metrices : ")
print("accuracy",metrics.accuracy_score(y_train, y_train_pred))
print("train recall : ", metrics.recall_score(y_train, y_train_pred,average='micro'))
print("train precision : ", metrics.precision_score(y_train, y_train_pred,average='micro'))
print("train f1_score : ", metrics.f1_score(y_train, y_train_pred,average='micro'))



# Testing Metrice by the help of confusion matrix

print("Testing Metrices : ")
print("accuracy",metrics.accuracy_score(y_test, y_test_pred))
print("test recall : ", metrics.recall_score(y_test, y_test_pred,average='micro'))
print("test precision : ", metrics.precision_score(y_test, y_test_pred,average='micro'))
print("test f1_score : ", metrics.f1_score(y_test, y_test_pred,average='micro'))




#importance of features

col_sorted_by_importance=model.feature_importances_.argsort()
feat_imp=pd.DataFrame({
    'cols':X_train_s.columns[col_sorted_by_importance],
    'imps':model.feature_importances_[col_sorted_by_importance]
})

import plotly_express as px
px.bar(feat_imp, x='cols', y='imps')



