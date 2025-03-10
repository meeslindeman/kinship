import egg.core as core
import torch.nn.functional as F
import torch
from archs.agents import (
    Sender, Receiver
)
from archs.lexicon_agent_wrapper import (
    LexiconSenderWrapper, LexiconReceiverWrapper,
    LexiconSenderReceiverRF,
    LexiconSenderReceiverGS
)
from options import Options

def get_game(opts: Options, num_node_features: int):

    def loss_nll(_sender_input, message, _receiver_input, receiver_output, labels, _aux_input):
        """
        NLL loss - differentiable and can be used with both GS and Reinforce
        """
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float().mean()

        if isinstance(message, tuple):
            alpha = 1.0
            message, commit_loss = message
            total_loss = nll + (alpha * commit_loss)
        else:
            total_loss = nll

        return total_loss, {"acc": acc}

    sender = Sender(num_node_features, opts)
    receiver = Receiver(num_node_features, opts)
    sender.to(opts.device)
    receiver.to(opts.device)

    if opts.max_len > 1:
        sender_wrapper = core.RnnSenderGS(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.embedding_size,
            hidden_size=opts.hidden_size,
            max_len=opts.max_len,
            temperature=opts.gs_tau,
            cell=opts.sender_cell
        )
        receiver_wrapper = core.RnnReceiverGS(
            agent=receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.embedding_size,
            hidden_size=opts.hidden_size,
            cell=opts.sender_cell
        )

        if opts.mode == 'gs':
            game = core.SenderReceiverRnnGS(sender_wrapper, receiver_wrapper, loss_nll)
        else:
            raise ValueError(f"Invalid wrapper: {opts.mode}")
    
    else:
        sender_wrapper = LexiconSenderWrapper(
            sender,
            opts.mode,
            opts.vocab_size,
            opts.hidden_size, 
            opts.gs_tau
        )
        receiver_wrapper = LexiconReceiverWrapper(
            receiver,
            opts.mode,
            opts.vocab_size, 
            opts.hidden_size
        )

        if opts.mode == 'continuous':
            game = LexiconSenderReceiverGS(sender_wrapper, receiver_wrapper, loss_nll)

        elif opts.mode == 'vq':
            game = LexiconSenderReceiverGS(sender_wrapper, receiver_wrapper, loss_nll)

        elif opts.mode == 'gs':
            game = LexiconSenderReceiverGS(sender_wrapper, receiver_wrapper, loss_nll)

        elif opts.mode == 'rf':
            game = LexiconSenderReceiverRF(
                sender_wrapper, receiver_wrapper, loss_nll,
                sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01
            )
        else:
            raise ValueError(f"Invalid wrapper: {opts.mode}")

    return game