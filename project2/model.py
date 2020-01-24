import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
      
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        
        """
        Parameters
        ----------
        embed_size = Size of embedding
        hidden_size = Number of nodes in the hidden layer
        vocab_size = The size of the output
        """
        # define the properties
        self.hidden_size = hidden_size #256 #512
        self.num_layers = num_layers    
        self.embed_size = embed_size #256  
        self.vocab_size = vocab_size #9955 
        
        
        # define LSTM
        self.lstm = nn.LSTM(input_size = self.embed_size, 
                            hidden_size = self.hidden_size , 
                            num_layers = self.num_layers,
                            batch_first =  True)
        
        # embedding layer
        self.embed = nn.Embedding(num_embeddings = self.vocab_size, 
                                  embedding_dim = self.embed_size)
        
        # output fully connected layer
        self.linear = nn.Linear(in_features = self.hidden_size, 
                               out_features = self.vocab_size) 
        
      
    def forward(self, features, captions):
        """
        Parameters:
        -----------
        Features = torch.Tensor | embedded features extracted from encoder
        captions = torch.Tensor | last batch of caption
        """
        # define the captions
        captions = captions[:, :-1]
        
        # embed the captions
        embeddings = self.embed(captions)
        
        features = features.unsqueeze(1)
        
        # concatenating embedded encoded features to embbedded and encoded caption for decoder
        embeddings = torch.cat((features, embeddings), dim=1)
        #[:, :-1,:]
        lstm_out , c = self.lstm(embeddings)
       
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        """
        - input: torch.Tensor, preprocessed and loaded image
        - states: touch.Tensor, hidden states
        - max_len: Maximum caption length
        
        """
        
        # initially empty predicted caption
        sentence = []
        
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = outputs.squeeze(1)
            outputs = self.linear(outputs)
            _, target = outputs.max(1)
            
            if target.item()==1:
                break
                
            sentence.append(target.item())
            inputs = self.embed(target.unsqueeze(1))
            
        return sentence
       