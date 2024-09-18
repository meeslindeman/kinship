import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.network import GAT, Transform
from archs.distractors import select_distractors

class Sender(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, layer, hidden_size, temperature):
        super(Sender, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size
        self.temp = temperature
        
        self.layer = Transform(self.num_node_features, embedding_size, heads) if layer == 'transform' else GAT(self.num_node_features, embedding_size, heads)
        self.fc = nn.Linear(embedding_size, hidden_size) 

    def forward(self, x, _aux_input):
        data = _aux_input

        batch_ptr, target_node_idx = data.ptr, data.target_node_idx

        h = self.layer(data)

        adjusted_target_node_idx = target_node_idx + batch_ptr[:-1]

        target_embedding = h[adjusted_target_node_idx]       

        output = self.fc(target_embedding)                           

        return output.view(-1, self.hidden_size)

class Receiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, layer, hidden_size, distractors):
        super(Receiver, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.distractors = distractors
        
        self.layer = Transform(self.num_node_features, embedding_size, heads) if layer == 'transform' else GAT(self.num_node_features, embedding_size, heads)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, _input, _aux_input):
        data = _aux_input

        h = self.layer(data)

        indices, _ = select_distractors(data, self.distractors)

        embeddings = h[indices]

        message = self.fc(message)

        embeddings = embeddings.view(data.num_graphs, (self.distractors + 1), -1)

        message = message.unsqueeze(2)  # Now [batch_size, 1, feature_size]

        dot_products = torch.bmm(embeddings, message).squeeze(-1)  # Result is [batch_size, num_distractors+1]

        probabilities = F.log_softmax(dot_products, dim=1)

        return probabilities
    
# =================================================================================================

class SenderRel(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, layer, hidden_size, temperature):
        super(SenderRel, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size
        self.temp = temperature
          
        self.layer = Transform(self.num_node_features, embedding_size, heads) if layer == 'transform' else GAT(self.num_node_features, embedding_size, heads) 
        self.fc = nn.Linear(2 * embedding_size, hidden_size) 

    def forward(self, x, _aux_input):
        data = _aux_input

        batch_ptr, target_node_idx, ego_idx = data.ptr, data.target_node_idx, data.ego_node_idx

        h = self.layer(data)

        adjusted_ego_idx = ego_idx + batch_ptr[:-1]
        adjusted_target_node_idx = target_node_idx + batch_ptr[:-1]
  
        target_embedding = torch.cat((h[adjusted_target_node_idx], h[adjusted_ego_idx]), dim=1) 

        output = self.fc(target_embedding)   

        return output # batch_size x hidden_size

class ReceiverRel(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, layer, hidden_size, distractors):
        super(ReceiverRel, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.distractors = distractors
        
        self.layer = Transform(self.num_node_features, embedding_size, heads) if layer == 'transform' else GAT(self.num_node_features, embedding_size, heads)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, _input, _aux_input):
        data = _aux_input
        h = self.layer(data)

        indices, _ = select_distractors(data, self.distractors)

        embeddings = h[indices]

        message = self.fc(message)

        embeddings = embeddings.view(data.num_graphs, (self.distractors + 1), -1)

        message = message.unsqueeze(2)  

        dot_products = torch.bmm(embeddings, message).squeeze(-1)  

        log_probabilities = F.log_softmax(dot_products, dim=1)
        
        return log_probabilities