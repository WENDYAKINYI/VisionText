# models.py

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the final FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # Shape: [B, 2048, 1, 1]

        if features.dim() == 4 and features.size(-1) == 1:
            features = features.squeeze(-1).squeeze(-1)  # Shape: [B, 2048]

        features = self.linear(features)                # Shape: [B, embed_size]
        features = self.bn(features)                    # BatchNorm1d expects [B, embed_size]
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])               # Remove <end> token
        features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        inputs = torch.cat((embeddings, features), dim=2)       # Shape: [B, T, embed+embed]
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(self.dropout(lstm_out))           # Shape: [B, T, vocab_size]
        return outputs

    def init_hidden_state(self, features):
        # Initialize with zeros if needed — can be enhanced with learned init
        batch_size = features.size(0)
        h = torch.zeros(1, batch_size, features.size(1)).to(features.device)
        c = torch.zeros(1, batch_size, features.size(1)).to(features.device)
        return h, c
