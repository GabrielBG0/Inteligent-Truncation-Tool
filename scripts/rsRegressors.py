import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import time

ds = pd.read_csv('../../CSVs/dataset_predictionsIntScore.csv', sep='§')

samples = ds.sample(frac=0.05, random_state=1)

data = ds['text'].tolist()
label = ds['score'].tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
X = vectorizer.fit_transform(data)

start = time.time()

mlpr = MLPRegressor()
mlpr_params = {'hidden_layer_sizes': [(100,), (100, 100), (100, 200), (100, 100, 100)],
               'alpha': [0.0001, 0.001, 0.01, 0.1],
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'learning_rate_init': [0.001, 0.01, 0.1, 0.5],
               'max_iter': [100, 200, 500, 1000],
               'tol': [0.0001, 0.001, 0.01, 0.1]}
rsMLPR = RandomizedSearchCV(mlpr, mlpr_params, n_iter=10,
                            cv=5, scoring='neg_mean_squared_error', random_state=1)

svr = SVR()
svr_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [0.001, 0.01, 0.1, 1, 10],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              'degree': [2, 3, 4, 5, 6],
              'coef0': [0.1, 0.2, 0.3, 0.4, 0.5],
              'shrinking': [True, False]}
rsSVR = RandomizedSearchCV(svr, svr_params, n_iter=10,
                           cv=5, scoring='neg_mean_squared_error', random_state=1)

rsMLPR.fit(X, label)
print('Best params MLPR:\n' + rsMLPR.best_params_ + '\n')
print('Best score MLPR:\n' + rsMLPR.best_score_ + '\n')

rsSVR.fit(X, label)
print('Best params SVR:\n' + rsSVR.best_params_ + '\n')
print('Best score SVR:\n' + rsSVR.best_score_ + '\n')

print('Time elapsed: ' + str(((time.time() - start) / 60 / 60)) + ' hours')

open('./results.txt', 'w').write(str(rsMLPR.best_params_) + '\n' + str(rsMLPR.best_score_) + '\n' + str(rsSVR.best_params_) +
                                 '\n' + str(rsSVR.best_score_) + '\n' + str(rsMLPR.best_estimator_) + '\n' + str(rsSVR.best_estimator_) + '\n' + 'Time elapsed: ' + str(((time.time() - start) / 60 / 60)) + ' hours')
