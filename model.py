import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as f

class EncoderCNN(nn.Module):
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use pretrained ResNet-50 weights
        res = models.resnet50(pretrained = True)
        modules = list(res.children())[:-1]
        self.res = nn.Sequential(*modules)
        self.embed = nn.Linear(res.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum = 0.01)
    
    def forward(self, images):
        # Extract feature vectors from image inputs
        with torch.no_grad():
            features = self.res(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # Decode image feature vectors to generate captions
        captions = captions [:, :-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        output = self.linear(hiddens)
        return output
    
    def sample_beam_search (self, inputs, states = None, max_len = 20, beam_width = 5):
        # Beam search approach
        # Accept a pre-processed image tensor and return top predicted sentences
        # Top sentences in indices and their corresponding inputs and states
        idx_sequences = [[[], 0.0, inputs, states]]
        
        for _ in range(max_len):
            all_candidates = []
            # Predict next word for each top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                log_p = f.log_softmax(outputs, -1)
                top_log_p, top_idx = log_p.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                for i in range(beam_width):
                    next_idx_seq, log_p = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_p += top_log_p[0][i].item()
                    inputs = self.embed(top_idx[i].unsqueeze(0).unsqueeze(0))
                    all_candidates.append([next_idx_seq, log_p, inputs, states])
            sort_candidates = sorted(all_candidates, key = lambda x: x[1], reverse = True)
            idx_sequences = sort_candidates[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]     