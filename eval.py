import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Load test data
with open('test_data.json', 'r') as f:
    test_data = json.load(f)

# Define the dataset class
class PathDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

        # Flatten the data
        self.items = []
        for label, paths in self.data.items():
            # Assuming label format like 'symptom_disease_positive'
            is_positive = 'positive' in label
            for path in paths:
                self.items.append((path, int(is_positive)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        inputs = self.tokenizer.encode_plus(
            path,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 1),
    torch.nn.Sigmoid()
)

# Load the trained model weights
model.load_state_dict(torch.load('bert_path_model.pth', map_location=torch.device('cpu')))
model.eval()

# Prepare DataLoader
test_loader = DataLoader(PathDataset(test_data, tokenizer), batch_size=16)

# Metrics evaluation
y_true = []
y_pred = []

for batch in test_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels'].numpy()
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.sigmoid(logits).detach().numpy()
    y_true.extend(labels)
    y_pred.extend(predictions)

y_pred = np.array(y_pred) >= 0.5  # Convert to binary predictions

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC Score: {roc_auc}")
