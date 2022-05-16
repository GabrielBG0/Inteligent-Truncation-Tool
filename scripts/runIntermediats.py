import joblib
import time
import pandas as pd

SIZE = 300

start = time.time()

mlpc = joblib.load('algos/mlpc.sav')
svc = joblib.load('algos/svc.sav')
rfc = joblib.load('algos/rfc.sav')
vectorizer = joblib.load('algos/vectorizer.sav')

dataset = open('datasetB.csv', 'r', encoding="utf8").read().split('\n')

for i in range(1, len(dataset) - 1):
    textV = list(dataset[i])
    textV[-2] = '§'
    dataset[i] = ''.join(textV)

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('§')

dataset.pop(0)
dataset.pop(-1)

data = [i[0] for i in dataset]
labels = [i[1] for i in dataset]

ds = pd.DataFrame({'text': [], 'MLPC': [],
                  'SVC': [], 'RFC': [], 'label': []})

for i in range(len(labels)):
    words = data[i].split(' ')
    words = [*words, *words[:SIZE]]
    windows = []
    for j in range(len(words) - SIZE):
        windows.append(' '.join(words[j:j + SIZE]))
    vec = vectorizer.transform(windows)
    predictionsMLPC = mlpc.predict(vec)
    predictionsSVC = svc.predict(vec)
    predictionsRFC = rfc.predict(vec)
    ds = pd.concat([ds, pd.DataFrame({
        'text': windows,
        'MLPC': predictionsMLPC,
        'SVC': predictionsSVC,
        'RFC': predictionsRFC,
        'label': [int(labels[i])]*len(windows)
    })])

ds.to_csv('dataset_predictionsInt.csv', index=False, sep='§')

print('Time: ' + str(((time.time() - start) / 60) / 60) + ' hours')
