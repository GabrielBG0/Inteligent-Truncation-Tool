from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = open('../../CSVs/datasetA.csv', 'r',
               encoding="utf8").read().split('\n')

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('§')

dataset.pop(0)
dataset[0][1]

data = [i[0] for i in dataset]
labels = [i[1] for i in dataset[:-1]]
data = data[:1200]
labels = labels[:1200]
print(data[0])
labels[0]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
X = vectorizer.fit_transform(data)
X.shape

# MLP, SVM, Random Forest

mlpc = MLPClassifier()
mlpcParams = {'hidden_layer_sizes': [(100,), (100, 100), (100, 200), (100, 100, 100)],
              'alpha': [0.0001, 0.001, 0.01, 0.1],
              'learning_rate': ['constant', 'invscaling', 'adaptive'],
              'learning_rate_init': [0.001, 0.01, 0.1, 0.5],
              'max_iter': [100, 200, 500, 1000],
              'tol': [0.0001, 0.001, 0.01, 0.1]}
gsMLPC = GridSearchCV(mlpc, mlpcParams, scoring='f1_macro', cv=5)


svc = SVC()
svcParams = {'C': [0.1, 1, 10, 100, 1000],
             'gamma': [0.001, 0.01, 0.1, 1, 10],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
             'degree': [2, 3, 4, 5, 6],
             'coef0': [0.1, 0.2, 0.3, 0.4, 0.5],
             'shrinking': [True, False]}
gsSVC = GridSearchCV(svc, svcParams, scoring='f1_macro', cv=5)


rfc = RandomForestClassifier()
rfcParams = {'n_estimators': [100, 200, 500],
             'criteion': ['gini', 'entropy'],
             'min_samples_split': [2, 5, 10, 20, 50, 100],
             'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
             'max_features': ['auto', 'sqrt', 'log2'],
             'max_depth': [None, 2, 5, 10, 20, 50, 100]}
gsRFC = GridSearchCV(rfc, rfcParams, scoring='f1_macro', cv=5)

gsMLPC.fit(X, labels)
gsSVC.fit(X, labels)
gsRFC.fit(X, labels)

print('Best params MLPC:\n' + gsMLPC.best_params_ + '\n')
print('Best params SVC: \n' + gsSVC.best_params_ + '\n')
print('Best params RFC: \n' + gsRFC.best_params_ + '\n')
