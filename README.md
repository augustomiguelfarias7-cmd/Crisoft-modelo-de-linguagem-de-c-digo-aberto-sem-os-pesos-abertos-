# CRIsoft - Modelo de Linguagem do Zero em Português Brasileiro
# Autor: Augusto

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ---------------------------
# 1. Configurações gerais
# ---------------------------
SEQ_LENGTH = 128
BATCH_SIZE = 4
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
EPOCHS = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# 2. Carregar dataset em português (streaming)
# ---------------------------
dataset = load_dataset("oscar", "unshuffled_deduplicated_pt", split="train", streaming=True)

# Pegar só alguns exemplos para teste
texts = []
for i, item in enumerate(dataset):
    texts.append(item["text"])
    if i >= 999:  # só 1000 textos para teste
        break

# ---------------------------
# 3. Criar vocabulário simples
# ---------------------------
from collections import Counter
import re

def tokenize(text):
    # simples: separar por espaço e pontuação
    tokens = re.findall(r"\w+|[^\s\w]", text.lower())
    return tokens

counter = Counter()
for text in texts:
    counter.update(tokenize(text))

vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(), start=0)}
vocab_size = len(vocab)
print(f"Tamanho do vocabulário: {vocab_size}")

# ---------------------------
# 4. Dataset custom
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length):
        self.data = []
        self.vocab = vocab
        self.seq_length = seq_length
        for text in texts:
            tokens = [vocab[t] for t in tokenize(text) if t in vocab]
            for i in range(0, len(tokens) - seq_length):
                self.data.append(tokens[i:i+seq_length+1])  # input + target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx][:-1])
        target = torch.tensor(self.data[idx][1:])
        return seq, target

train_dataset = TextDataset(texts, vocab, SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------
# 5. Modelo Transformer do Zero
# ---------------------------
class CRISoftModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

model = CRISoftModel(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LENGTH).to(device)

# ---------------------------
# 6. Treinamento
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for batch_idx, (seq, target) in enumerate(train_loader):
        seq, target = seq.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

# ---------------------------
# 7. Salvar modelo
# ---------------------------
torch.save(model.state_dict(), "crissoft_zero.pt")
print("Treinamento do CRIsoft do zero finalizado! 🎉")
