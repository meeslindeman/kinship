import egg.core as core
import torch.nn.functional as F
import torch
from archs.agents import (
    Sender, Receiver,
    SenderRel, ReceiverRel
)
from archs.lexicon_agent_wrapper import (
    LexiconSenderWrapper, LexiconReceiverWrapper,
    LexiconSenderReceiverRF,
    LexiconSenderReceiverGS
)
from options import Options

def get_game(opts: Options, num_node_features: int):

    def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        """
        NLL loss - differentiable and can be used with both GS and Reinforce
        """
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float().mean()
        return nll, {"acc": acc}

    if opts.set_up == 'single':
        sender = Sender(num_node_features=num_node_features,
                        embedding_size=opts.embedding_size,
                        heads=opts.heads,
                        layer=opts.layer,
                        hidden_size=opts.hidden_size,
                        temperature=opts.gs_tau)

        receiver = Receiver(num_node_features=num_node_features,
                            embedding_size=opts.embedding_size,
                            heads=opts.heads, layer=opts.layer,
                            hidden_size=opts.hidden_size,
                            distractors=opts.distractors)

    elif opts.set_up == 'relationship':
        sender = SenderRel(num_node_features=num_node_features,
                           embedding_size=opts.embedding_size,
                           heads=opts.heads, layer=opts.layer,
                           hidden_size=opts.hidden_size,
                           temperature=opts.gs_tau)

        receiver = ReceiverRel(num_node_features=num_node_features,
                               embedding_size=opts.embedding_size,
                               heads=opts.heads, layer=opts.layer,
                               hidden_size=opts.hidden_size,
                               distractors=opts.distractors)
    else:
        raise ValueError(f"Invalid set_up: {opts.set_up}")

    sender_wrapper = LexiconSenderWrapper(
        sender,
        opts.mode,
        opts.vocab_size, opts.hidden_size
    )
    receiver_wrapper = LexiconReceiverWrapper(
        receiver,
        opts.mode,
        opts.vocab_size, opts.hidden_size
    )

    if opts.mode == 'continuous':
        game = LexiconSenderReceiverGS(sender_wrapper, receiver_wrapper, loss_nll)

    elif opts.mode == 'gs':
        game = LexiconSenderReceiverGS(sender_wrapper, receiver_wrapper, loss_nll)

    elif opts.mode == 'rf':
        game = LexiconSenderReceiverRF(
            sender_wrapper, receiver_wrapper, loss_nll,
            sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01
        )
    else:
        raise ValueError(f"Invalid wrapper: {opts.wrapper}")

    return game
