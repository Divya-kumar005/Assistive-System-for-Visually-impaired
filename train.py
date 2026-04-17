import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from Models.encoder import EncoderCNN
from Models.decoder import DecoderRNN
from utils.data_loader import FlickrDataset
from utils.tokeniser import clean_caption, create_tokenizer

# --------------------
# Load captions
# --------------------
def load_captions(file_path):
    captions = {}
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            img_name = tokens[0].split('#')[0]
            caption = clean_caption(tokens[1])

            if img_name not in captions:
                captions[img_name] = []
            captions[img_name].append(caption)
    return captions

captions_dict = load_captions("data/captions.txt")

# Flatten captions
all_captions = [cap for caps in captions_dict.values() for cap in caps]

# Tokenizer
tokenizer = create_tokenizer(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = 20

# Dataset
dataset = FlickrDataset("data/images", captions_dict, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Models
encoder = EncoderCNN()
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=vocab_size)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# --------------------
# Training Loop
# --------------------
num_epochs = 5

for epoch in range(num_epochs):
    for images, captions in dataloader:
        features = encoder(images)

        outputs = decoder(features, captions[:, :-1])
        targets = captions[:, 1:]

        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save models
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")