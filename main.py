import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Amaya\Downloads\anemia.csv")
print(df.head(5))

print(df.info())

print(df.shape)

results = df["Result"].value_counts()
results.plot(kind = 'bar', color = ['green', 'red'])
plt.xlabel("Result")
plt.ylabel("Frequency")
plt.title("Count of Result")
plt.show()

# Exploratory Data Analysis.

print(df.describe())

sns.countplot(data=df, x='Result', hue='Result', palette='Set2', legend=False)
plt.title("Anemia Class Distribution")
plt.xlabel("Anemic (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

sns.countplot(data=df, x='Gender', hue='Result', palette='coolwarm')
plt.title("Anemia Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

output = df["Gender"].value_counts()
output.plot(kind = 'bar', color = ["blue","orange"])
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Gender Count")
plt.show()

sns.displot(df['Hemoglobin'], kde = True)

plt.figure(figsize=(6,6))
ax = sns.barplot(y=df['Hemoglobin'], x=df['Gender'], hue=df['Result'], ci=None)
ax.set(xlabel=['male','female'])
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title("Mean Hemoglobin by Gender and Result")
plt.show()

sns.pairplot(df)


sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)  #It gives values between -1 and 1, where:
fig = plt.gcf()                                                    #+1 = perfect positive correlation
fig.set_size_inches(10, 8)                                         #-1 = perfect negative correlation
plt.show()                                                         #0 = no correlation

features = df.drop('Result', axis=1)
print(features)

target = df['Result']
print(target)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=20)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test,y_pred)
c_lr = classification_report(y_test,y_pred)
print("Accuracy Score:", acc_lr)
print(c_lr)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_rf = accuracy_score(y_test,y_pred)
c_rf = classification_report(y_test,y_pred)
print("Accuracy Score:", acc_rf)
print(c_rf)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_dt = accuracy_score(y_test,y_pred)
c_dt = classification_report(y_test,y_pred)
print("Accuracy Score:", acc_dt)
print(c_dt)

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)

acc_NB = accuracy_score(y_test,y_pred)
c_NB = classification_report(y_test,y_pred)
print("Accuracy Score:", acc_NB)
print(c_NB)

from sklearn.svm import SVC

support_vector = SVC()
support_vector.fit(x_train, y_train)
y_pred = support_vector.predict(x_test)

acc_svc = accuracy_score(y_test,y_pred)
c_svc = classification_report(y_test,y_pred)
print("Accuracy Score:", acc_svc)
print(c_svc)

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
y_pred = GBC.predict(x_test)

acc_gbc = accuracy_score(y_test,y_pred)
c_gbc = classification_report(y_test,y_pred)
print("Accuracy Score:", acc_gbc)
print(c_gbc)

prediction = support_vector.predict([[0,11.6,22.3,30.9,74.5]])
prediction[0]

if prediction[0] == 0:
    print("You Don't have any Anemia Disease")
elif prediction[0] == 1:
    print("You have Anemia Disease")

model = pd.DataFrame({'Model':['LogisticRegression','RandomForestClassifier','DecisionTreeClassifier',
                               'GaussianNB','SVC','GradientBoostingClassifier'],
                     'Score':[acc_lr,acc_rf,acc_dt,acc_NB,acc_svc,acc_gbc],
                     })
print(model)

import pickle
import warnings

pickle.dump(GBC,open("model.pkl","wb"))
