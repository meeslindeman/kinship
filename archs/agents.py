import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from options import Options
from archs.network import GAT, Transform
from archs.distractors import select_distractors

class Sender(nn.Module):
    def __init__(self, num_node_features: int, opts: Options):
        super(Sender, self).__init__()
        self.layer = (
            Transform(num_node_features, opts.embedding_size, opts.heads)
            if opts.layer == 'transform'
            else GAT(num_node_features, opts.embedding_size, opts.heads)
        )
        self.fc = nn.Linear(2 * opts.embedding_size, opts.hidden_size)

        self.use_vq = opts.with_vq
        if self.use_vq:
            self.vq_layer = VectorQuantize(
                dim=opts.hidden_size, 
                codebook_size=opts.codebook_size, 
                decay=0.8
            ) 
        
    def forward(self, x, _aux_input, finetune: bool=False):
        data = _aux_input
        batch_ptr, target_node_idx, ego_idx = data.ptr, data.target_node_idx, data.ego_node_idx
        if finetune:
            with torch.no_grad():
                h = self.layer(data)
        else:
            h = self.layer(data)

        adjusted_ego_idx = ego_idx + batch_ptr[:-1]
        adjusted_target_node_idx = target_node_idx + batch_ptr[:-1]
        target_embedding = torch.cat((h[adjusted_target_node_idx], h[adjusted_ego_idx]), dim=1)
        output = self.fc(target_embedding)

        if self.use_vq:
            output, _, commit_loss = self.vq_layer(output)
            return output, commit_loss
        return output, None # batch_size x hidden_size

class Receiver(nn.Module):
    def __init__(self, num_node_features: int, opts: Options):
        super(Receiver, self).__init__()
        self.distractors = opts.distractors
        self.layer = (
            Transform(num_node_features, opts.embedding_size, opts.heads)
            if opts.layer == 'transform'
            else GAT(num_node_features, opts.embedding_size, opts.heads)
        )
        self.fc = nn.Linear(opts.hidden_size, opts.embedding_size)

    def forward(self, message, _input, _aux_input, finetune: bool=False):
        data = _aux_input

        if finetune:
            with torch.no_grad():
                h = self.layer(data)
        else:
            h = self.layer(data)

        indices, _ = select_distractors(
            data,
            self.distractors if not getattr(data, 'evaluation', False) else len(data.target_node) - 1,
            evaluation=getattr(data, 'evaluation', False)
        )

        embeddings = h[indices]

        batch_size = data.num_graphs
        num_candidates = embeddings.size(0) // batch_size

        embeddings = embeddings.view(batch_size, num_candidates, -1)
        message = self.fc(message)
        message = message.unsqueeze(2)

        dot_products = torch.bmm(embeddings, message).squeeze(-1)

        # break tie
        diff = dot_products.abs() - dot_products.abs().max()
        eps = (diff < 1e-10).float() * dot_products.abs().max() * 1e-5 * torch.randn_like(dot_products)
        dot_products = dot_products + eps
        log_probabilities = F.log_softmax(dot_products, dim=1)

        if True: # not self.training:
            diff = (log_probabilities - log_probabilities.max()).abs()
            diff = (diff < 1e-10).float().sum(-1).mean().item()
            if diff > 1:
                print(diff)

        return log_probabilities

