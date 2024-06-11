import egg.core as core
import torch.nn.functional as F
from archs.agents import Sender, Receiver, SenderRel, ReceiverRel
from options import Options

def get_game(opts: Options, num_node_features: int):
    """
    Returns a game object based on the given options and number of node features.
    
    Args:
        opts (Options): The options for the game.
        num_node_features (int): The number of features for each node.
    
    Returns:
        game (SenderReceiverRnnGS): The game object.
    """

    def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        """
        NLL loss - differentiable and can be used with both GS and Reinforce
        """
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float().mean()
        return nll, {"acc": acc}

    if opts.set_up == 'single':
        sender = Sender(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, temperature=opts.gs_tau) 
        receiver = Receiver(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, distractors=opts.distractors)
    elif opts.set_up == 'relationship':
        sender = SenderRel(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, temperature=opts.gs_tau) 
        receiver = ReceiverRel(num_node_features=num_node_features, embedding_size=opts.embedding_size, heads=opts.heads, hidden_size=opts.hidden_size, distractors=opts.distractors)
    else:
        raise ValueError(f"Invalid set_up: {opts.set_up}")

    if opts.mode == 'gs':
        sender_gs = core.RnnSenderGS(sender, 
                                    opts.vocab_size, 
                                    opts.embedding_size, 
                                    opts.hidden_size, 
                                    max_len=opts.max_len, 
                                    temperature=opts.gs_tau, 
                                    cell=opts.sender_cell)
        
        receiver_gs = core.RnnReceiverGS(receiver, 
                                        opts.vocab_size, 
                                        opts.embedding_size, 
                                        opts.hidden_size, 
                                        cell=opts.sender_cell)
        
        game = core.SenderReceiverRnnGS(sender_gs, receiver_gs, loss_nll)

    elif opts.mode == 'rf':
        sender_reinforce = core.RnnSenderReinforce(sender, 
                                                opts.vocab_size, 
                                                opts.embedding_size, 
                                                opts.hidden_size, 
                                                max_len=opts.max_len, 
                                                cell=opts.sender_cell)
        
        receiver_reinforce = core.RnnReceiverDeterministic(receiver, 
                                                    opts.vocab_size, 
                                                    opts.embedding_size, 
                                                    opts.hidden_size, 
                                                    cell=opts.sender_cell)
        
        game = core.SenderReceiverRnnReinforce(sender_reinforce, receiver_reinforce, loss_nll, sender_entropy_coeff=0.01, receiver_entropy_coeff=0)
    else:
        raise ValueError(f"Invalid wrapper: {opts.wrapper}")

    return game

