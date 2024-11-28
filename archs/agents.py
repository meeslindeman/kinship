import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from options import Options
from archs.network import GAT, Transform, RGCN
from archs.distractors import select_distractors


class Sender(nn.Module):
    def __init__(self, num_node_features: int, opts: Options):
        super(Sender, self).__init__()
        if opts.layer == 'transform':
            self.layer = Transform(num_node_features, opts.embedding_size, opts.heads)
        elif opts.layer == 'gat':
            self.layer = GAT(num_node_features, opts.embedding_size, opts.heads)
        elif opts.layer == 'rgcn':
            self.layer = RGCN(num_node_features, opts.embedding_size) # add num relations based on data? or manually?
        else:
            raise ValueError("Unsupported layer type: Choose from 'transform', 'gat', or 'rgcn'")
        
        self.fc = nn.Linear(2 * opts.embedding_size, opts.hidden_size)

        self.vq = opts.mode == 'vq'
        if self.vq:
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

        if self.vq:
            output, _, commit_loss = self.vq_layer(output)
            return output, commit_loss
        return output, None # batch_size x hidden_size

class Receiver(nn.Module):
    def __init__(self, num_node_features: int, opts: Options, layer: nn.Module=None):
        super(Receiver, self).__init__()
        self.distractors = opts.distractors
        if layer: # use shared graph nn
            self.layer = layer
        else:
            if opts.layer == 'transform':
                self.layer = Transform(num_node_features, opts.embedding_size, opts.heads)
            elif opts.layer == 'gat':
                self.layer = GAT(num_node_features, opts.embedding_size, opts.heads)
            elif opts.layer == 'rgcn':
                self.layer = RGCN(num_node_features, opts.embedding_size)
            else:
                raise ValueError("Unsupported layer type: Choose from 'transform', 'gat', or 'rgcn'")
            
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

        embeddings = h
        batch_size = data.num_graphs
        num_candidates = embeddings.size(0) // batch_size

        embeddings = embeddings.view(batch_size, num_candidates, -1)
        message = self.fc(message)
        message = message.unsqueeze(2)
        dot_products = torch.bmm(embeddings, message).squeeze(-1)

        log_probabilities = F.log_softmax(dot_products, dim=1)

        # elimintate all nodes that are neither distractors or target
        mask = torch.zeros(batch_size * num_candidates)
        mask[indices] = 1
        mask = mask.view(batch_size, -1)
        log_probabilities = log_probabilities - (1 - mask) * 1e5
        return log_probabilities

