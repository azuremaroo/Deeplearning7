# https://www.kaggle.com/competitions/leaf-classification/rules
# 99가지 종류의 이미지 데이터를 수치화
# kaggle Leaf Classification 페이지 내 Late Submission 버튼을 통해 sub mission 파일을 제출해 경쟁 참여 가능

# 퀴즈 : classifier showdown 코드를 복사해 동작하도록 수정
# https://www.kaggle.com/code/jeffd23/10-classifier-showdown-in-scikit-learn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

train = pd.read_csv('leaf/train.csv')
test = pd.read_csv('leaf/test.csv')


# Swiss army knife function to organize the data
def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1) # drop(, axis=1) : 컬럼 삭제
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes

# x_train, y_train, x_test
train, labels, test, test_ids, classes = encode(train, test)
print(train.head(1))

# 더이상 사용할 수 없는 코드
# sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
# for train_index, test_index in sss:
#     X_train, X_test = train.values[train_index], train.values[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]

split = StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=23)

for train_index, test_index in split.split(train, labels):
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    # GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    print("=" * 30)
    print(name)

    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))

    # train_predictions = clf.predict_proba(X_test)
    # ll = log_loss(y_test, train_predictions)
    # print("Log Loss: {}".format(ll))

    # log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    # log = log.append(log_entry)

print("=" * 30)


# sns.set_color_codes("muted")
# sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
#
# plt.xlabel('Accuracy %')
# plt.title('Classifier Accuracy')
# plt.show()
#
# sns.set_color_codes("muted")
# sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
#
# plt.xlabel('Log Loss')
# plt.title('Classifier Log Loss')
# plt.show()

# Predict Test Set
favorite_clf = LinearDiscriminantAnalysis()
favorite_clf.fit(X_train, y_train)
test_predictions = favorite_clf.predict_proba(test)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('leaf/submission.csv', index = False)
print(submission.tail())

