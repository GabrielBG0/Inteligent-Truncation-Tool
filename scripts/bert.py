import pandas as pd
from datasets import load_metric
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import torch
import copy
from transformers import BertTokenizerFast
import warnings


warnings.filterwarnings('ignore')


MAX_TRAIN_LENTH = 128


class FNNDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(model, eval_dataloader, metric):
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()


def truncate(encodings, method="h", max_len=512, p_heads=0.5, p_tails=0.5):
    aux = copy.deepcopy(encodings)

    if method == "t":
        # for encoding in aux["input_ids"]:
        for i in range(len(aux["input_ids"])):
            if len(aux["input_ids"][i]) > max_len:
                aux_input_ids = [101, *aux["input_ids"][i][-max_len+1:]]
                aux["input_ids"][i] = aux_input_ids.copy()
            else:
                aux["input_ids"][i] = [*aux["input_ids"][i],
                                       *[0]*(max_len - len(aux["input_ids"][i]))]
        for i in range(len(aux["token_type_ids"])):
            if len(aux["token_type_ids"][i]) > max_len:
                aux_token_type_ids = [aux["token_type_ids"][i]
                                      [0], *aux["token_type_ids"][i][-max_len+1:]]
                aux["token_type_ids"][i] = aux_token_type_ids.copy()
            else:
                aux["token_type_ids"][i] = [*aux["token_type_ids"][i],
                                            *[0]*(max_len - len(aux["token_type_ids"][i]))]
        for i in range(len(aux["attention_mask"])):
            if len(aux["attention_mask"][i]) > max_len:
                aux_attention_mask = [aux["attention_mask"][i]
                                      [0], *aux["attention_mask"][i][-max_len+1:]]
                aux["attention_mask"][i] = aux_attention_mask.copy()
            else:
                aux["attention_mask"][i] = [*aux["attention_mask"][i],
                                            *[0]*(max_len - len(aux["attention_mask"][i]))]

    elif method == "h":
        for i in range(len(aux["input_ids"])):
            if len(aux["input_ids"][i]) > max_len:
                aux_input_ids = [*aux["input_ids"][i][:max_len-1], 102]
                aux["input_ids"][i] = aux_input_ids.copy()
            else:
                aux["input_ids"][i] = [*aux["input_ids"][i],
                                       *[0]*(max_len - len(aux["input_ids"][i]))]
        for i in range(len(aux["token_type_ids"])):
            if len(aux["token_type_ids"][i]) > max_len:
                aux_token_type_ids = [*aux["token_type_ids"][i]
                                      [:max_len-1], aux["token_type_ids"][i][-1]]
                aux["token_type_ids"][i] = aux_token_type_ids.copy()
            else:
                aux["token_type_ids"][i] = [*aux["token_type_ids"][i],
                                            *[0]*(max_len - len(aux["token_type_ids"][i]))]
        for i in range(len(aux["attention_mask"])):
            if len(aux["attention_mask"][i]) > max_len:
                aux_attention_mask = [*aux["attention_mask"][i]
                                      [:max_len-1], aux["attention_mask"][i][-1]]
                aux["attention_mask"][i] = aux_attention_mask.copy()
            else:
                aux["attention_mask"][i] = [*aux["attention_mask"][i],
                                            *[0]*(max_len - len(aux["attention_mask"][i]))]

    elif method == "ht":
        head_len = int(max_len * p_heads)
        tail_len = int(max_len * p_tails)
        for i in range(len(aux["input_ids"])):
            if len(aux["input_ids"][i]) > max_len:
                aux_input_ids = [*aux["input_ids"][i][:head_len],
                                 *aux["input_ids"][i][-tail_len:]]
                aux["input_ids"][i] = aux_input_ids.copy()
            else:
                aux["input_ids"][i] = [*aux["input_ids"][i],
                                       *[0]*(max_len - len(aux["input_ids"][i]))]
        for i in range(len(aux["token_type_ids"])):
            if len(aux["token_type_ids"][i]) > max_len:
                aux_token_type_ids = [*aux["token_type_ids"][i]
                                      [:head_len], *aux["token_type_ids"][i][-tail_len:]]
                aux["token_type_ids"][i] = aux_token_type_ids.copy()
            else:
                aux["token_type_ids"][i] = [*aux["token_type_ids"][i],
                                            *[0]*(max_len - len(aux["token_type_ids"][i]))]
        for i in range(len(aux["attention_mask"])):
            if len(aux["attention_mask"][i]) > max_len:
                aux_attention_mask = [*aux["attention_mask"][i]
                                      [:head_len], *aux["attention_mask"][i][-tail_len:]]
                aux["attention_mask"][i] = aux_attention_mask.copy()
            else:
                aux["attention_mask"][i] = [*aux["attention_mask"][i],
                                            *[0]*(max_len - len(aux["attention_mask"][i]))]

    return aux


train_texts_ds = pd.read_csv('../FakeNewsNetCSV/train_texts.csv')
test_texts_ds = pd.read_csv('../FakeNewsNetCSV/test_texts.csv')

train_labels = train_texts_ds['label'].values.tolist()
test_labels = test_texts_ds['label'].values.tolist()

train_texts = train_texts_ds['text'].values.tolist()
test_texts = test_texts_ds['text'].values.tolist()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_labels_vec = []
train_encodings_vec = []
t_len = len(train_texts)

for i in range(5):
    if i != 4:
        train_encodings_vec.append(
            tokenizer(train_texts[i * int(t_len / 5):(i + 1) * int(t_len / 5)], padding=False, verbose=False))
        train_labels_vec.append(
            train_labels[i * int(t_len / 5):(i + 1) * int(t_len / 5)])
    else:
        train_encodings_vec.append(
            tokenizer(train_texts[i * int(t_len / 5):], padding=False, verbose=False))
        train_labels_vec.append(
            train_labels[i * int(t_len / 5):])

test_encodings = tokenizer(test_texts, padding=False, verbose=False)

print('Starting Tails Trainig...')
# 1

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)

test_encodings_t = truncate(test_encodings, method="t", max_len=512)

test_dataset = FNNDataset(test_encodings_t, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=16)

for train_encodings in train_encodings_vec:

    train_encodings_t = truncate(
        train_encodings, method="t", max_len=MAX_TRAIN_LENTH)

    train_dataset = FNNDataset(train_encodings_t, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(16):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

model.eval()

print('trainig has finished.')
print('final evaluation for Tails is: ')

whantedStatistics = ['accuracy', 'precision', 'recall', 'f1']
for statistic in whantedStatistics:
    metric = load_metric(statistic, verbose=False)
    print(statistic + ': ' +
          str(compute_metrics(model, test_dataloader, metric)[statistic]))

print('')
print('Starting Heads Traning...')
# 2

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)

test_encodings_t = truncate(test_encodings, method="h", max_len=512)

test_dataset = FNNDataset(train_encodings_t, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=16)

for train_encodings in train_encodings_vec:

    train_encodings_t = truncate(
        train_encodings, method="h", max_len=MAX_TRAIN_LENTH)

    train_dataset = FNNDataset(train_encodings_t, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(16):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

model.eval()

print('trainig has finished.')
print('final evaluation for Heads is: ')

whantedStatistics = ['accuracy', 'precision', 'recall', 'f1']
for statistic in whantedStatistics:
    metric = load_metric(statistic, verbose=False)
    print(statistic + ': ' +
          str(compute_metrics(model, test_dataloader, metric)[statistic]))

print('')
print('Starting H&T 0.25/0.75 Traning...')
# 3

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)
optim = AdamW(model.parameters(), lr=5e-5)

test_encodings_t = truncate(
    test_encodings, method="ht", max_len=512, p_heads=0.25, p_tails=0.75)

test_dataset = FNNDataset(train_encodings_t, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=16)

for train_encodings in train_encodings_vec:

    train_encodings_t = truncate(
        train_encodings, method="ht", max_len=MAX_TRAIN_LENTH, p_heads=0.25, p_tails=0.75)

    train_dataset = FNNDataset(train_encodings_t, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(16):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

model.eval()

print('trainig has finished.')
print('final evaluation for H&T 0.25/0.75 is: ')

whantedStatistics = ['accuracy', 'precision', 'recall', 'f1']
for statistic in whantedStatistics:
    metric = load_metric(statistic, verbose=False)
    print(statistic + ': ' +
          str(compute_metrics(model, test_dataloader, metric)[statistic]))

print('')
print('Starting H&T 0.5/0.5 Traning...')
# 4

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)

test_encodings_t = truncate(
    test_encodings, method="ht", max_len=512, p_heads=0.25, p_tails=0.75)

test_dataset = FNNDataset(train_encodings_t, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=16)

for train_encodings in train_encodings_vec:

    train_encodings_t = truncate(
        train_encodings, method="ht", max_len=MAX_TRAIN_LENTH, p_heads=0.5, p_tails=0.5)

    train_dataset = FNNDataset(train_encodings_t, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(16):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

model.eval()

print('trainig has finished.')
print('final evaluation for H&T 0.5/0.5 is: ')

whantedStatistics = ['accuracy', 'precision', 'recall', 'f1']
for statistic in whantedStatistics:
    metric = load_metric(statistic, verbose=False)
    print(statistic + ': ' +
          str(compute_metrics(model, test_dataloader, metric)[statistic]))

print('')

print('Starting H&T 0.75/0.25 Traning...')
# 5

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

optim = AdamW(model.parameters(), lr=5e-5)

test_encodings_t = truncate(
    test_encodings, method="ht", max_len=512, p_heads=0.25, p_tails=0.75)

test_dataset = FNNDataset(train_encodings_t, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=16)

for train_encodings in train_encodings_vec:

    train_encodings_t = truncate(
        train_encodings, method="ht", max_len=MAX_TRAIN_LENTH, p_heads=0.75, p_tails=0.25)

    train_dataset = FNNDataset(train_encodings_t, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(16):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

model.eval()

print('trainig has finished.')
print('final evaluation for H&T 0.75/0.25 is: ')

whantedStatistics = ['accuracy', 'precision', 'recall', 'f1']
for statistic in whantedStatistics:
    metric = load_metric(statistic, verbose=False)
    print(statistic + ': ' +
          str(compute_metrics(model, test_dataloader, metric)[statistic]))

print('')
