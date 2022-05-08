from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import joblib

start = time.time()

dataset = open('CSVs/datasetA.csv', 'r',
               encoding="utf8").read().split('\n')

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('§')

dataset.pop(0)

data = [i[0] for i in dataset]
labels = [i[1] for i in dataset[:-1]]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
X = vectorizer.fit_transform(data)

mlpc = MLPClassifier(alpha=0.001, hidden_layer_sizes=(
    100, 100, 100), learning_rate='invscaling', learning_rate_init=0.01, max_iter=1000)

svc = SVC(C=10, degree=3, gamma=1, kernel='sigmoid')

rfc = RandomForestClassifier(n_estimators=500, min_samples_split=20,
                             min_samples_leaf=2, max_features='auto', max_depth=50, criterion='gini')

mlpc.fit(X, labels)
joblib.dump(mlpc, 'mlpc.sav')

svc.fit(X, labels)
joblib.dump(svc, 'svc.sav')

rfc.fit(X, labels)
joblib.dump(rfc, 'rfc.sav')

print('Time: ' + str(((time.time() - start) / 60) / 60) + ' hours')
