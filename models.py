# models.py 

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()
        features = self.linear(features)
        features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions, object_vec=None):
        embeddings = self.embed(captions[:, :-1])
        if object_vec is not None:
            if object_vec.shape[-1] == features.shape[-1]:
                features = features + object_vec
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(self.dropout(lstm_out))
        return outputs

    def init_hidden_state(self, encoder_out_mean):
        batch_size = encoder_out_mean.size(0)
        h = torch.zeros((1, batch_size, self.lstm.hidden_size)).to(encoder_out_mean.device)
        c = torch.zeros((1, batch_size, self.lstm.hidden_size)).to(encoder_out_mean.device)
        return h, c
