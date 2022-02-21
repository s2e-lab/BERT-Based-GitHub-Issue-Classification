# %% [markdown]
# We are using PyTorch as our deep learning framework. 
# Importing necessary libraries to pre-processing, tokenizing and evaluation.

# %%
import random
from torch.utils.data import DataLoader, SequentialSampler
import torch

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification


# %%
import os
import numpy as np
import pandas as pd
import gensim
from sklearn.metrics import f1_score, precision_score, recall_score


# %% [markdown]
# Checking the device. We will proceed if there is a GPU available.

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
else:
    exit(0)

# %% [markdown]
# Download the test set and load into a DataFrame.

# %%
# if not os.path.isfile("./Dataset/github-labels-top3-803k-test.csv"):
#     !curl "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-test.tar.gz" | tar -xz 
#     !mv github-labels-top3-803k-train.csv ./Dataset/

testdf = pd.read_csv("./Dataset/github-labels-top3-803k-test.csv")

# %% [markdown]
# Use the same label map used in the training.

# %%
label_dict = {'bug': 0, 'enhancement': 1, 'question': 2}
testdf['label'] = testdf.issue_label.replace(label_dict)

# %% [markdown]
# Pre-preocessing function for removing whitespace and creating new feature.

# %%
def preprocess(row):
    # concatenate title and body, then remove whitespaces
    doc = ""
    doc += str(row.issue_title)
    doc += " "
    doc += str(row.issue_body)
    # https://radimrehurek.com/gensim/parsing/preprocessing.html
    doc = gensim.parsing.preprocessing.strip_multiple_whitespaces(doc)
    return doc

# %% [markdown]
# Applying preporcessing step on the dataframe.

# %%
testdf['issue_data'] = testdf.apply(preprocess, axis=1)

newTestDF = testdf[['issue_label', 'issue_data', 'label']]
testdf = newTestDF.copy()
print(testdf.head())

# %% [markdown]
# Initiating tokenizer and encoding the test set.

# %%

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)


encoded_data_test = tokenizer.batch_encode_plus(
    testdf.issue_data.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='longest',
    truncation=True,
    return_tensors='pt'
)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(testdf.label.values)

# %% [markdown]
# Creating TensorDataset from encoded and masked test and creating a dataloader for testing.

# %%
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

batch_size = 4


dataloader_test = DataLoader(dataset_test,
                             sampler=SequentialSampler(dataset_test),
                             batch_size=batch_size)

# %% [markdown]
# Declaring function for result generation.

# %%

def result_generation(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

        P_c = precision_score(labels_flat, preds_flat, average=None, labels=[label])[0]
        R_c = recall_score(labels_flat, preds_flat, average=None, labels=[label])[0]
        F1_c = f1_score(labels_flat, preds_flat, average=None, labels=[label])[0]

        print(f"=*= {label_dict_inverse[label]} =*=")
        # print("Full precision:\t",P_c)
        # print("Full recall:\t\t",R_c)
        # print("Full F1 score:\t",F1_c)
        print(f"precision:\t{P_c:.4f}")
        print(f"recall:\t\t{R_c:.4f}")
        print(f"F1 score:\t{F1_c:.4f}")
        print()

    P = precision_score(labels_flat, preds_flat, average='micro')
    R = recall_score(labels_flat, preds_flat, average='micro')
    F1 = f1_score(labels_flat, preds_flat, average='micro')

    print("=*= global =*=")
    print(f"precision:\t{P:.4f}")
    print(f"recall:\t\t{R:.4f}")
    print(f"F1 score:\t{F1:.4f}")
    print()

# %% [markdown]
# Fixing seed value for random sampling.

# %%

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# %% [markdown]
# Declaring the function for evaluating.

# %%
def evaluate(model, dataloader_val):
    
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


# %% [markdown]
# Evaluating the model on different model states.

# %%
for i in range(1, 5):
    print("Epoch: ", i)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(
                                                              label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False)

    model.to(device)
    model.load_state_dict(torch.load(
        './Models/finetuned_BERT_epoch_'+str(i)+'.model', map_location=device))

    # %%
    _, predictions, true_vals = evaluate(model, dataloader_test)

    # %%
    result_generation(predictions, true_vals)


