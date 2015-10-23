# need Python 3.4.3 :: Anaconda 2.3.0 (x86_64)
# This submission is based on the guide on https://www.dataquest.io/section/kaggle-competitions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import operator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation as CV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# reading in the train and test csv files.
titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")


## =====================Preprocessing the data===========================

def get_title(name):
    '''
    This method uses regex to parse through name and find the
    Passenger's title.
    '''
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

family_id_mapping = {}
def get_family_id(row):
    '''
    This method assigns a family id based on last name + family size, and save
    the assignment in family_id_mapping.
    This method can only be used after row["FamilySize"] is created.
    '''
    last_name = row["Name"].split(",")[0]

    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key = operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Title_mapping assignes an id to all avaliable titles. Some titles are
# rare, so they are grouped with the same id based on similarity.

title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Dr":5, "Rev":6, "Major":7,
    "Col":7, "Mlle":8, "Mme":8, "Don":9, "Lady":10, "Countess":10, "Jonkheer":10, "Sir":9, "Capt":7, "Ms":2, "Dona":10}

# actual Preprocessing take place.
for samples in (titanic, titanic_test):
    # missing values are replaced with median.
    samples["Age"] = samples["Age"].fillna(samples["Age"].median())
    samples["Fare"] = samples["Fare"].fillna(samples["Fare"].median())
    # convert sex into numeric id.
    samples.loc[samples["Sex"] == "male", "Sex"] = 0.0
    samples.loc[samples["Sex"] == "female", "Sex"] = 1.0

    # replace missing embarked location with the most frequent entry "S",
    # and then convert to a numeric id.
    samples["Embarked"] = samples["Embarked"].fillna("S")
    samples.loc[samples["Embarked"] == "S", "Embarked"] = 0.0
    samples.loc[samples["Embarked"] == "C", "Embarked"] = 1.0
    samples.loc[samples["Embarked"] == "Q", "Embarked"] = 2.0

    # create "FamilySize" and "NameLength" features.
    samples["FamilySize"] = samples["SibSp"] + samples["Parch"]
    samples["NameLength"] = samples["Name"].apply(lambda x: len(x))

    # create "Title" features using get_title method.
    titles = samples["Name"].apply(get_title)
    for k, v in title_mapping.items():
        titles[titles == k] = v
    samples["Title"] = titles

    # create "FamilyId" features using get_family_id method.
    family_ids = samples.apply(get_family_id, axis = 1)
    family_ids[titanic["FamilySize"] < 3] = -1
    samples["FamilyId"] = family_ids

## ================ Feature Analysis ==============================
#pick the predictors for analysis.
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize",  "Title"]

# using selectKBest to examine the most important features and plot the result.
selector = SelectKBest(f_classif, k = 5)
selector.fit(titanic[predictors], titanic["Survived"])
scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation = 45 )
plt.show()


## ================= Training and Prediction ======================
# Adding the prediction of linear regression as a new feature "LinReg"
alg1 = LinearRegression()
alg1.fit(titanic[predictors], titanic["Survived"])
titanic["LinReg"] = alg1.predict(titanic[predictors])
titanic_test["LinReg"] = alg1.predict(titanic_test[predictors])

# initializing three algorithms.
alg2 = LogisticRegression(random_state = 1)
alg3 = RandomForestClassifier(random_state = 1, n_estimators = 100, min_samples_split = 4, min_samples_leaf =2)
alg4 = GradientBoostingClassifier(random_state = 1, n_estimators = 20, max_depth = 3)

# Run all three predictor and place the result in the list: predictions
predictions = []
for alg, alg_name in [(alg2, "logistic regression"), (alg3, "random forest"), (alg4, "gradient boosting")]:
    scores = CV.cross_val_score(alg, titanic[predictors + ["LinReg"] ], titanic["Survived"], cv = 4)
    print("The accuracy of %s model is: %1.4f" % (alg_name, scores.mean()))
    alg.fit(titanic[predictors + ["LinReg"]], titanic["Survived"])
    predictions.append(alg.predict(titanic_test[predictors + ["LinReg"]]))

# Use majority voting rule to determine the final result.
final_predictions = (predictions[0] + predictions[1] + predictions[2])
final_predictions[final_predictions <   2] = 0
final_predictions[final_predictions >=  2] = 1
final_predictions = final_predictions.astype(int)

# save the result for submission.
submission = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": final_predictions
})

submission.to_csv("kaggle.csv", index = False)
