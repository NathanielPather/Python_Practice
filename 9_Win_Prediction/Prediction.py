
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset
url = "dataset.csv"

names = [
    'win'
    'firstBlood',
    'firstTower',
    'T1_P1_kills',
    'T1_P1_deaths',
    'T1_P1_totalDamageDealtToChampions',
    'T1_P1_damageDealtToObjectives',
    'T1_P1_damageDealtToTurrets',
    'T1_P1_timeCCingOthers',
    'T1_P1_totalMinionsKilleds',
    'T1_P1_wardsPlaced',
    'T1_P1_wardsKilled',
    'T1_P2_kills',
    'T1_P2_deaths',
    'T1_P2_totalDamageDealtToChampions',
    'T1_P2_damageDealtToObjectives',
    'T1_P2_damageDealtToTurrets',
    'T1_P2_timeCCingOthers',
    'T1_P2_totalMinionsKilleds',
    'T1_P2_wardsPlaced',
    'T1_P2_wardsKilled',
    'T1_P3_kills',
    'T1_P3_deaths',
    'T1_P3_totalDamageDealtToChampions',
    'T1_P3_damageDealtToObjectives',
    'T1_P3_damageDealtToTurrets',
    'T1_P3_timeCCingOthers',
    'T1_P3_totalMinionsKilleds',
    'T1_P3_wardsPlaced',
    'T1_P3_wardsKilled',
    'T1_P4_kills',
    'T1_P4_deaths',
    'T1_P4_totalDamageDealtToChampions',
    'T1_P4_damageDealtToObjectives',
    'T1_P4_damageDealtToTurrets',
    'T1_P4_timeCCingOthers',
    'T1_P4_totalMinionsKilleds',
    'T1_P4_wardsPlaced',
    'T1_P4_wardsKilled',
    'T1_P5_kills',
    'T1_P5_deaths',
    'T1_P5_totalDamageDealtToChampions',
    'T1_P5_damageDealtToObjectives',
    'T1_P5_damageDealtToTurrets',
    'T1_P5_timeCCingOthers',
    'T1_P5_totalMinionsKilleds',
    'T1_P5_wardsPlaced',
    'T1_P5_wardsKilled',
    'T2_P1_kills',
    'T2_P1_deaths',
    'T2_P1_totalDamageDealtToChampions',
    'T2_P1_damageDealtToObjectives',
    'T2_P1_damageDealtToTurrets',
    'T2_P1_timeCCingOthers',
    'T2_P1_totalMinionsKilleds',
    'T2_P1_wardsPlaced',
    'T2_P1_wardsKilled',
    'T2_P2_kills',
    'T2_P2_deaths',
    'T2_P2_totalDamageDealtToChampions',
    'T2_P2_damageDealtToObjectives',
    'T2_P2_damageDealtToTurrets',
    'T2_P2_timeCCingOthers',
    'T2_P2_totalMinionsKilleds',
    'T2_P2_wardsPlaced',
    'T2_P2_wardsKilled',
    'T2_P3_kills',
    'T2_P3_deaths',
    'T2_P3_totalDamageDealtToChampions',
    'T2_P3_damageDealtToObjectives',
    'T2_P3_damageDealtToTurrets',
    'T2_P3_timeCCingOthers',
    'T2_P3_totalMinionsKilleds',
    'T2_P3_wardsPlaced',
    'T2_P3_wardsKilled',
    'T2_P4_kills',
    'T2_P4_deaths',
    'T2_P4_totalDamageDealtToChampions',
    'T2_P4_damageDealtToObjectives',
    'T2_P4_damageDealtToTurrets',
    'T2_P4_timeCCingOthers',
    'T2_P4_totalMinionsKilleds',
    'T2_P4_wardsPlaced',
    'T2_P4_wardsKilled',
    'T2_P5_kills',
    'T2_P5_deaths',
    'T2_P5_totalDamageDealtToChampions',
    'T2_P5_damageDealtToObjectives',
    'T2_P5_damageDealtToTurrets',
    'T2_P5_timeCCingOthers',
    'T2_P5_totalMinionsKilleds',
    'T2_P5_wardsPlaced',
    'T2_P5_wardsKilled'
]

dataset = read_csv(url, names=names)

array = dataset.values

X = array[:, 2:93]
Y = array[:, 1]
# Y = Y.astype('int')
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=1
)

models = []
models.append(
    (
        'LR',
        LogisticRegression(
            solver='liblinear',
            multi_class='ovr'
        )
    )
)
models.append(
    (
        'LDA',
        LinearDiscriminantAnalysis()
    )
)
models.append(
    (
        'KNN',
        KNeighborsClassifier()
    )
)
models.append(
    (
        'CART',
        DecisionTreeClassifier()
    )
)
models.append(
    (
        'NB',
        GaussianNB()
    )
)
models.append(
    (
        'SVM',
        SVC(gamma='auto')
    )
)

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(
        n_splits=2000000000,
        random_state=1,
        shuffle=True
    )
    cv_results = cross_val_score(model, X_train, Y_train)
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(
   results,
   labels=names
)
pyplot.title('Algorithm Comparison')
pyplot.show()

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
