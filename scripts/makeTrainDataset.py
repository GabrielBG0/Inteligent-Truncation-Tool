import pandas as pd
import gc

mainDs = pd.z({'text': [], 'score': []})

for i in range(5):
    ds = pd.read_csv(
        'CSVs/prediction_int_score_part/{index}.csv'.format(index=i), sep='§').sample(60000)
    mainDs = pd.concat([mainDs, ds], ignore_index=True)
    del ds
    gc.collect()

mainDs.to_csv('CSVs/trainDS300k.csv', sep='§', index=False)
