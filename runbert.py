import torch
from torch.utils.data import DataLoader
from alive_progress import alive_bar
import time
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SIZE = 300
MAX_TRAIN_LENTH = 400

dataset = open('dataset.csv', 'r', encoding="utf8").read().split('\n')

for i in range(1, len(dataset) - 1):
    textV = list(dataset[i])
    textV[-2] = '§'
    dataset[i] = ''.join(textV)

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('§')

dataset.pop(0)

data = [i[0] for i in dataset]
labels = [i[1] for i in dataset[:-1]]

print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')

tokenizer = AutoTokenizer.from_pretrained('fake-news-heatmap-creator')
model = AutoModelForSequenceClassification.from_pretrained('fake-news-heatmap-creator')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(device)

ds = pd.DataFrame({'text':[], 'prediction':[], 'label':[]})
with alive_bar(len(labels), title='Generating Dataset') as bar:
  for i in range(len(labels)):
    bar.text = f'-> Analizing news N {i}, please wait...'
    words = data[i].split(' ')
    words = [*words, *words[:SIZE]]
    windows = []
    for j in range(len(words) - SIZE):
      windows.append(' '.join(words[j:j + SIZE])) 

    # Colocar um for com 300 linhas
    x = 300
    final_list= lambda windows, SIZE: [windows[k:k+SIZE] for k in range(0, len(windows), SIZE)]
    windowsBatch=final_list(windows, SIZE)
    for b in windowsBatch:
      batch = tokenizer(b, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_TRAIN_LENTH)
      with torch.no_grad():
        batch.to(device)
        outputs = model(**batch)
        predictions = F.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        
        ds = pd.concat([ds,pd.DataFrame({
            'text':b, 
            'prediction':predictions.cpu(), 
            'label':[int(labels[i])]*len(b)
            })
          ])
    bar()


ds.to_csv('dataset_predictions.csv', index=False)