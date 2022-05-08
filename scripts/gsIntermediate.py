from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time

start = time.time()

dataset = open('CSVs/datasetA.csv', 'r',
               encoding="utf8").read().split('\n')

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('§')

dataset.pop(0)

data = [i[0] for i in dataset]
labels = [i[1] for i in dataset[:-1]]
data = data[:1200]
labels = labels[:1200]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
X = vectorizer.fit_transform(data)

# MLP, SVM, Random Forest

mlpc = MLPClassifier()
mlpcParams = {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
              'alpha': [0.0001, 0.001, 0.01],
              'learning_rate': ['constant', 'invscaling', 'adaptive'],
              'learning_rate_init': [0.001, 0.01],
              'max_iter': [1000]}
gsMLPC = GridSearchCV(mlpc, mlpcParams, scoring='f1_macro', cv=5)


svc = SVC()
svcParams = {'C': [0.1, 1, 10, 100],
             'gamma': [0.001, 0.01, 0.1, 1],
             'kernel': ['linear', 'rbf', 'sigmoid'],
             'degree': [3, 4, 5]}
gsSVC = GridSearchCV(svc, svcParams, scoring='f1_macro', cv=5)


rfc = RandomForestClassifier()
rfcParams = {'n_estimators': [100, 200, 500],
             'criterion': ['gini', 'entropy'],
             'min_samples_split': [2, 5, 10, 20, 50, 100],
             'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
             'max_features': ['auto', 'sqrt', 'log2'],
             'max_depth': [None, 2, 5, 10, 20, 50, 100]}
gsRFC = RandomizedSearchCV(rfc, rfcParams, scoring='f1_macro', cv=5)

gsMLPC.fit(X, labels)
print('Best params MLPC: \n' + str(gsMLPC.best_params_) + '\n')
print('Best score MLPC: \n' + str(gsMLPC.best_score_) + '\n')

gsSVC.fit(X, labels)
print('Best params SVC: \n' + str(gsSVC.best_params_) + '\n')
print('Best score SVC: \n' + str(gsSVC.best_score_) + '\n')

gsRFC.fit(X, labels)
print('Best params RFC: \n' + str(gsRFC.best_params_))
print('Best score RFC: \n' + str(gsRFC.best_score_) + '\n')

print('Time: ' + str(((time.time() - start) / 60) / 60) + ' hours')
