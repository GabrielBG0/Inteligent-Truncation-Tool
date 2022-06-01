import pandas as pd

df = pd.read_csv('CSVs/dataset_predictionsIntScore.csv',
                 delimiter='§', encoding='utf-8')

dfLen = len(df['score'])

for i in range(5):
    if i != 4:
        df.iloc[i * int(dfLen / 5):(i + 1) * int(dfLen / 5),
                :].to_csv('CSVs/prediction int score part/' + str(i) + '.csv', sep='§', index=False)
    else:
        df.iloc[i * int(dfLen / 5):, :].to_csv('CSVs/prediction int score part/' +
                                               str(i) + '.csv', sep='§', index=False)
