import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import time
import joblib
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()

ds = pd.read_csv('dataset_predictionsIntScore.csv', sep='§')

data = ds['text'].tolist()
label = ds['score'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.3, random_state=7722)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)

vectorizer.fit(data)

joblib.dump(vectorizer, 'vect.sav')

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print('starting MLP trainig...')

mlpr = MLPRegressor(tol=0.001, max_iter=1000, learning_rate_init=0.1,
                    learning_rate='adaptive', hidden_layer_sizes=(100, 200), alpha=0.0001)

mlpr.fit(X_train, y_train)

joblib.dump(mlpr, 'mlpr.sav')

y_pred = mlpr.predict(X_test)

print('R2 score for MLPR: ' + str(r2_score(y_test, y_pred)))
print('Mean absolute percentage error score for MLPR: ' +
      str(mean_absolute_percentage_error(y_test, y_pred)) + '\n')


print('MLPR traning completed in: ' +
      str(((time.time() - start_time) / 60 / 60)) + ' hours' + '\n')

svr_start_time = time.time()

print('starting SVR trainig...' + '\n')

svm = SVR(shrinking=False, kernel='poly', gamma=1, degree=3, coef0=0.4, C=1)

svm.fit(X_train, y_train)

joblib.dump(svm, 'svr.sav')

y_pred = svm.predict(X_test)

print('R2 score for SVR: ' + str(r2_score(y_test, y_pred)))
print('Mean absolute percentage error score for SVR: ' +
      str(mean_absolute_percentage_error(y_test, y_pred)) + '\n')


print('SVR traning completed in: ' +
      str(((time.time() - svr_start_time) / 60 / 60)) + ' hours' + '\n')

print('total time was: ' + str(((time.time() - start_time) / 60 / 60)) + ' hours')
