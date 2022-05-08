from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

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

rfc = RandomForestClassifier()
rfcParams = {'n_estimators': [100, 200, 500],
             'criterion': ['gini', 'entropy'],
             'min_samples_split': [2, 5, 10, 20, 50, 100],
             'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100],
             'max_features': ['auto', 'sqrt', 'log2'],
             'max_depth': [None, 2, 5, 10, 20, 50, 100]}
gsRFC = RandomizedSearchCV(rfc, rfcParams, scoring='f1_macro', cv=5)

gsRFC.fit(X, labels)
print('Best params RFC: \n' + str(gsRFC.best_params_))
print('Best score RFC: \n' + str(gsRFC.best_score_) + '\n')
