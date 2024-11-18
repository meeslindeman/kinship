import torch
import egg.core as core
import numpy as np
import torch.nn.functional as F
import collections
from torch.distributions import Categorical
from itertools import product
from need_probs import get_need_probs
from collections import Counter, defaultdict
from math import log2


class ResultsCollector(core.Callback):
    def __init__(self, options, game, eval_loader, **kwargs):
        self.options = options
        self.game = game
        self.eval_loader = eval_loader
        self.results = []

        self.print_train_loss = kwargs.get('print_train_loss', True)

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        train_metrics = self._aggregate_metrics(loss, logs, "train", epoch)

        if epoch % self.options.evaluation_interval == 0:
            if self.options.evaluation:
                eval_logs = self.evaluate(epoch)
                train_metrics['eval_acc'] = eval_logs['accuracy']
                train_metrics['evaluation'] = eval_logs['evaluation']
                train_metrics['complexity'] = self._complexity(eval_logs['evaluation'])
                train_metrics['information_loss'] = self._information_loss(eval_logs['evaluation'])

            if self.options.messages:
                messages = self._messages_to_indices(logs.message)
                target_nodes = [target_node for batch in logs.aux_input for target_node in batch['target_node']]

                per_target_message_counts = self._compute_message_counts(messages, target_nodes)
                train_metrics["message_counts"] = per_target_message_counts

        self.results.append(train_metrics)
        self._print_to_console({k: v for k, v in train_metrics.items() if k not in ['message_counts', 'evaluation']})

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        test_metrics = self._aggregate_metrics(loss, logs, "test", epoch)
        self.results.append(test_metrics)
        self._print_to_console({k: v for k, v in test_metrics.items()})

    def _aggregate_metrics(self, loss: float, logs: core.Interaction, mode: str, epoch: int) -> dict:
        metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        return {
            "epoch": epoch,
            "mode": mode,
            "loss": loss,
            **metrics
        }

    def _messages_to_indices(self, messages_tensor):
        if self.options.mode == "rf":
            return messages_tensor.tolist()
        else:
            return [message.argmax(dim=-1).tolist() for message in messages_tensor]

    def _compute_message_counts(self, messages, target_nodes):
        per_target_message_counts = collections.defaultdict(collections.Counter)
        for message, target_node in zip(messages, target_nodes):
            message_str = str(message)
            per_target_message_counts[target_node][message_str] += 1
        return dict(per_target_message_counts)

    def _print_to_console(self, metrics: dict):
        output_message = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        print(output_message, flush=True)

    def evaluate(self, epoch):
        self.game.eval()
        total_correct = 0
        total_samples = 0
        per_target_data = []

        with torch.no_grad():
            for batch in self.eval_loader:
                # Unpack the batch
                sender_input, labels, receiver_input, aux_input = batch
                device = self.options.device
                sender_input = sender_input.to(device)
                labels = labels.to(device)
                aux_input = aux_input.to(device)
                aux_input.evaluation = True

                # Forward pass
                loss, interaction = self.game.forward(sender_input, labels, receiver_input, aux_input)

                # Get message
                if self.options.mode == "rf":
                    message = interaction.message.tolist()
                else:
                    message = interaction.message.argmax(dim=-1).tolist()

                # Get the receiver output logits and compute probabilities
                receiver_probs = F.softmax(interaction.receiver_output, dim=-1)

                # Check if the prediction is correct
                predicted_label = receiver_probs.argmax().item()
                correct = (predicted_label == 0)  # Target is at index 0

                total_correct += int(correct)
                total_samples += 1

                # Collect data per target
                per_target_data.append({
                    'target_node': aux_input.target_node[0],
                    'message': message,
                    'receiver_output': receiver_probs.cpu().numpy().tolist(),
                    'predicted_label': predicted_label,
                    'correct': correct,
                    'epoch': epoch
                })

        accuracy = total_correct / total_samples
        evaluation_results = {
            'epoch': epoch,
            'accuracy': accuracy,
            'evaluation': per_target_data
        }
        return evaluation_results

    def _complexity(self, counts):
        need_probs = get_need_probs('dutch')
        normalized_need_probs = {target: prob / sum(need_probs.values()) for target, prob in need_probs.items()}

        targets = [element['target_node'] for element in counts]
        messages = [tuple(element['message']) for element in counts]

        count_target = defaultdict(float)  # for p(u) equivalent
        count_msg_target = defaultdict(lambda: defaultdict(float))  # for p(w|u)
        count_target_msg = defaultdict(lambda: defaultdict(float))  # for p(u|w)

        for target, message in zip(targets, messages):
            count_target[target] += normalized_need_probs[target]
            count_msg_target[target][message] += 1
            count_target_msg[message][target] += normalized_need_probs[target]

        p_target = defaultdict(float, {target: normalized_need_probs[target] for target in count_target.keys()})
        p_message_given_target = defaultdict(lambda: defaultdict(float), {
            target: defaultdict(float, {
                message: count_msg_target[target][message] / sum(count_msg_target[target].values())
                for message in count_msg_target[target].keys()
            })
            for target in count_msg_target.keys()
        })

        p_target_given_message = defaultdict(lambda: defaultdict(float), {
            message: defaultdict(float, {
                target: count_target_msg[message][target] / sum(count_target_msg[message].values())
                for target in count_target_msg[message].keys()
            })
            for message in count_target_msg.keys()
        })

        complexity = 0
        for target, message in product(targets, messages):
            complexity += (p_target[target] * p_message_given_target[target][message] *
                        log2((p_target_given_message[message][target] + 1e-10) / p_target[target]))

        return complexity


    def _information_loss(self, counts):
        targets = [element['target_node'] for element in counts]
        receiver_outputs = [output for element in counts for output in element['receiver_output']]

        need_probs = get_need_probs('dutch')
        normalized_need_probs = {target: prob / sum(need_probs.values()) for target, prob in need_probs.items()}

        information_loss = 0

        for i, target in enumerate(targets):
            receiver_output = log2(receiver_outputs[i][0])
            target_prob = normalized_need_probs[target]

            cross_entropy = -target_prob * receiver_output

            information_loss += cross_entropy

        return information_loss

    def get_results(self):
        return self.results