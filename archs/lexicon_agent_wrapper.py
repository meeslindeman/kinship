import torch
from torch import nn
from egg.core.gs_wrappers import gumbel_softmax_sample
from typing import Optional
from egg.core.interaction import LoggingStrategy
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from torch.distributions import Categorical
from egg.core.baselines import Baseline, MeanBaseline
from collections import defaultdict

class LexiconSenderWrapper(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        agent_type: str,  # continuous, gs, rf
        vocab_size: Optional[int],
        hidden_size: Optional[int],
        gs_tau: float = 1.0
    ):
        super().__init__()
        self.agent = agent
        self.agent_type = agent_type
        self.gs_tau = gs_tau

        self.vocab_size = vocab_size

        if agent_type in ['rf', 'gs']:
            self.vocab_size = vocab_size
            self.lex_f = nn.Linear(hidden_size, vocab_size)
        if agent_type == 'gs': 
            self.gs_tau = gs_tau

    def forward(self, x, aux_input=None, warm_up: bool=True):
        output = self.agent(x, aux_input, finetune=not warm_up)

        if self.agent_type == 'vq':
            h, loss = output
            return h, loss

        h, _ = output
        if self.agent_type == 'continuous':
            return h

        if self.agent_type == 'gs':
            lex_logit = self.lex_f(h)

            if warm_up:
                if self.training:
                    return F.softmax(lex_logit / 0.1)
                else:
                    output = lex_logit.argmax(dim=-1)
                    return F.one_hot(output, self.vocab_size).float()
            else:
                return gumbel_softmax_sample(
                    lex_logit, self.gs_tau, self.training, False
                )

        elif self.agent_type == 'rf':
            lex_logit = self.lex_f(h)
            lex_logit = F.log_softmax(lex_logit, dim=1)
            distr = Categorical(logits=lex_logit)
            entropy = distr.entropy()

            if self.training:
                output = distr.sample()
                logit = distr.log_prob(output)

                explore_chance = torch.rand(output.shape)
                explore_output = torch.randint(low=0, high=self.vocab_size, size=output.shape)
                explore_entropy = torch.zeros_like(entropy)
                explore_logit = torch.zeros_like(logit)

                explore_p = 0
                output = torch.where(
                    explore_chance >= explore_p,
                    output,
                    explore_output
                )
                entropy = torch.where(
                    explore_chance >= explore_p,
                    entropy,
                    explore_entropy
                )
                logit = torch.where(
                    explore_chance >= explore_p,
                    logit,
                    explore_logit
                )
            else:
                output = lex_logit.argmax(dim=1)
                logit = distr.log_prob(output)
            return output, logit, entropy

        else:
            not NotImplementedError()

class LexiconReceiverWrapper(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        agent_type: str,  # continuous, gs, rf
        vocab_size: Optional[int],
        hidden_size: Optional[int],
    ):
        super().__init__()
        self.agent = agent
        self.agent_type = agent_type

        if agent_type in ['vq', 'gs', 'rf']:
            self.vocab_size = vocab_size
            self.lex_f = nn.Linear(vocab_size, hidden_size)

    def forward(self, message, input=None, aux_input=None, warm_up: bool=True):
        if self.agent_type == 'vq':
            message = message[0]
            message = self.lex_f(message.float())
            return self.agent(message, input, aux_input, finetune=not warm_up)

        if self.agent_type == 'continuous':
            return self.agent(message, input, aux_input, finetune=not warm_up)

        elif self.agent_type == 'gs':
            message = self.lex_f(message)
            return self.agent(message, input, aux_input, finetune=not warm_up)

        elif self.agent_type == 'rf':
            message = F.one_hot(message, num_classes=self.vocab_size)
            message = self.lex_f(message.float())
            output = self.agent(message, input, aux_input, finetune=not warm_up)
            logits = torch.zeros(output.size(0)).to(output.device)
            entropy = logits
            return output, logits, entropy


class LexiconSenderReceiverGS(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: callable,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )
        self.count = 0
        self.warm_up = True

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        # if self.count == 10000:
        #     self.warm_up = False
        #     print('finetune')
        # self.count += 1
        self.warm_up = False

        message = self.sender(sender_input, aux_input, warm_up=self.warm_up)
        receiver_output = self.receiver(message, receiver_input, aux_input, warm_up=self.warm_up)
        loss, aux = self.loss(
            sender_input,
            message,
            receiver_input,
            receiver_output,
            labels,
            aux_input,
        )

        aux_info = {}
        for name, value in aux.items():
            aux_info[name] = value.unsqueeze(0)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        if isinstance(message, tuple):
            message = message[0]

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(1),
            aux=aux_info,
        )

        return loss.mean(), interaction


class LexiconSenderReceiverRF(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.baselines = defaultdict(baseline_type)
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )


    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input, aux_input)
        receiver_output, log_prob_r, entropy_r = self.receiver(
            message, receiver_input, aux_input
        )

        loss, aux = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        aux_info = {}
        for name, value in aux.items():
            aux_info[name] = value.unsqueeze(0)

        weighted_entropy = (
            entropy_s.mean() * self.sender_entropy_coeff
            + entropy_r.mean() * self.receiver_entropy_coeff
        )

        log_prob = log_prob_s + log_prob_r

        policy_loss = (
            (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
        ).mean()

        optimized_loss = policy_loss - weighted_entropy
        optimized_loss += loss.mean()

        if self.training:
            self.baselines["loss"].update(loss)

        aux_info["sender_entropy"] = entropy_s.detach()
        aux_info["receiver_entropy"] = entropy_r.detach()

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=torch.ones(1),
            aux=aux_info,
        )

        return optimized_loss, interaction