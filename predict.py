import torch
from PIL import Image
from torchvision import transforms

from .Models.encoder import EncoderCNN
from .Models.decoder import DecoderRNN

# Load tokenizer manually (IMPORTANT)
import pickle
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

vocab_size = len(tokenizer.word_index) + 1
max_length = 20

# Load models
encoder = EncoderCNN()
decoder = DecoderRNN(256, 512, vocab_size)

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))

encoder.eval()
decoder.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    result = ['startseq']

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([result])[0]
        seq = torch.tensor(seq).unsqueeze(0)

        features = encoder(image)
        output = decoder(features, seq)

        predicted = output.argmax(2)[:, -1].item()
        word = tokenizer.index_word.get(predicted)

        if word is None or word == 'endseq':
            break

        result.append(word)

    return ' '.join(result[1:])

# Test
print(generate_caption("data/images/sample.jpg"))