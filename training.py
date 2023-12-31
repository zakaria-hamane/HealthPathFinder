import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch


# Load data
with open('train_data.json', 'r') as f:
    train_data = json.load(f)
with open('test_data.json', 'r') as f:
    test_data = json.load(f)


# Prepare the dataset
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


from transformers import BertForSequenceClassification

# Initialize BERT model with a sigmoid output layer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Modify the output layer to sigmoid
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 1),
    torch.nn.Sigmoid()
)

from torch.optim import AdamW
from tqdm import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader = DataLoader(PathDataset(train_data, tokenizer), batch_size=16, shuffle=True)
test_loader = DataLoader(PathDataset(test_data, tokenizer), batch_size=16)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
# Use the test_loader to evaluate the model

torch.save(model.state_dict(), 'bert_path_model.pth')
