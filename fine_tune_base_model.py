import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AdamW


path_to_json = 'config.json'

f = open(path_to_json, encoding="utf8")     
config = json.load(f)
 
# Paths and base model
base_model = config['base_model']
custom_cache_directory = config['cache_dir']
dataset_path = config['dataset_path']

# Hyperparameters
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']

# Other
use_seq2seq = config['use_seq2seq']


start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom dataset from Excel into a DataFrame.
df = pd.read_excel(dataset_path)

# Tokenize the dataset.
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=custom_cache_directory)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenized_data = tokenizer(df['Prompt'].tolist(), return_tensors='pt', padding=True, truncation=True)

# Add labels to tokenized_data (assuming you have a 'Response' column in your Excel).
tokenized_data['labels'] = tokenizer(df['Response'].tolist(), return_tensors='pt', padding=True, truncation=True)['input_ids']


# ----------------------------------- Dataset  ------------------------------------

# Define a custom PyTorch Dataset.
class GAIADataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.tokenized_data.items()}


dataset = GAIADataset(tokenized_data)
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

# Create Subset objects for train and validation sets.
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create DataLoader instances for train and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------------

if use_seq2seq:
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, cache_dir=custom_cache_directory)
else:
    model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=custom_cache_directory)
model.to(device)

#quit()

# Set all parameters to require gradients.  Fine tune every layer of the pre-trained model.
for param in model.parameters():
    param.requires_grad = True

''' 
 Set up training parameters
 1e-5 = 1 x 10 to the -1 = 0.00001
 Lower value, better for more epochs. Higher - less epochs.
'''

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for batch in train_dataloader:
        # Move batch to the same device as the model.
        batch = {key: value.to(device) for key, value in batch.items()}
    
        inputs = {key: batch[key] for key in batch if key != 'labels'}
        
        # Forward pass.
        outputs = model(**inputs, labels=batch['labels'])
        loss = outputs.loss
                      
        # Backward pass.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Validate
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to the same device as the model.
            batch = {key: value.to(device) for key, value in batch.items()}
        
            inputs = {key: batch[key] for key in batch if key != 'labels'}
            
            # Forward pass
            outputs = model(**inputs, labels=batch['labels'])
            loss = outputs.loss
            
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    
    # Early stopping.
    if avg_val_loss > avg_train_loss:
        break
        
    # Print status for each epoch.
    print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


# Save
model.save_pretrained('GAIA400M')
tokenizer.save_pretrained('GAIA400M')

end = time.time()
print('Time (mins): ', (end - start) / 60)