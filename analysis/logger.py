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
from graph.kemp_build import NODES

class ResultsCollector(core.Callback):
    def __init__(self, options, game, eval_loader, **kwargs):
        self.options = options
        self.game = game
        self.eval_loader = eval_loader
        self.results = []
        self.language = options.language
        self.natural_language_profile = self._load_natural_languge_profile(self.language)
        self.calc_topsim = core.TopographicSimilarity(
            sender_input_distance_fn="edit",
            message_distance_fn="edit",
            compute_topsim_train_set=False,
            compute_topsim_test_set=True,
            is_gumbel=True
        )

        self.print_train_loss = kwargs.get('print_train_loss', True)

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        train_metrics = self._aggregate_metrics(loss, logs, "train", epoch)

        if epoch % self.options.evaluation_interval == 0:
            if self.options.evaluation:
                eval_logs = self.evaluate(epoch)
                train_metrics['evaluation'] = eval_logs['evaluation']
                train_metrics['eval_acc'] = eval_logs['accuracy']

                for ego in ['Bob', 'Alice']:
                    results = self._compute_complexity_infoloss_accuracy_emerged_language(eval_logs['evaluation'], ego)
                    train_metrics[f'eval_acc@{self.language}@{ego}'] = results['accuracy']
                    train_metrics[f'complexity@{self.language}@{ego}'] = results['complexity']
                    train_metrics[f'information_loss@{self.language}@{ego}'] = results['information_loss']

            if self.options.messages:
                messages = self._messages_to_indices(logs.message)
                target_nodes = [target_node for batch in logs.aux_input for target_node in batch['target_node']]

                per_target_message_counts = self._compute_message_counts(messages, target_nodes)
                train_metrics["message_counts"] = per_target_message_counts

        self.results.append(train_metrics)
        self._print_to_console({k: v for k, v in train_metrics.items() if k not in ['message_counts', 'evaluation']})

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        test_metrics = self._aggregate_metrics(loss, logs, "test", epoch)
        # leave out for now
        # topsim = self.calc_topsim.compute_topsim(
        #     torch.flatten(logs.sender_input, start_dim=1), 
        #     logs.message.argmax(dim=-1) if self.topsim_calculator.is_gumbel else logs.message
        # )
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

                # Get the receiver output log_prop and compute probabilities
                receiver_probs = interaction.receiver_output.exp()

                # Check if the prediction is correct
                predicted_label = receiver_probs.argmax().item()
                correct = (predicted_label == labels.item())

                total_correct += int(correct)
                total_samples += 1

                assert NODES[predicted_label] != 'Ego', f"{receiver_probs[0, predicted_label]}"

                # Collect data per target
                per_target_data.append({
                    'ego_node': aux_input.ego_node[0],
                    'target_node_idx': aux_input.target_node_idx[0].item(),
                    'target_node': aux_input.target_node[0],
                    'message': message,
                    'receiver_output': receiver_probs.cpu().numpy().tolist(),
                    'predicted_label': NODES[predicted_label],
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

    def _load_natural_languge_profile(self, language):
        return {
            'need_probs': get_need_probs(language),
            'target_message': None
        }

    def _estimate_prob_given_count(self, target_message):
        p_target = self.natural_language_profile['need_probs']
        targets = [element['target'] for element in target_message]
        messages = [tuple(element['message']) for element in target_message]

        count_target = defaultdict(lambda: 1e-10)  # for p(u) equivalent
        count_msg_target = defaultdict(lambda: defaultdict(lambda: 1e-10))  # for p(w|u)
        count_target_msg = defaultdict(lambda: defaultdict(lambda: 1e-10))  # for p(u|w)

        for target, message in zip(targets, messages):
            count_target[target] += p_target[target]
            count_msg_target[target][message] += 1
            count_target_msg[message][target] += p_target[target]

        p_message_given_target = defaultdict(lambda: defaultdict(lambda: 1e-10), {
            target: defaultdict(lambda: 1e-10, {
                message: count_msg_target[target][message] / sum(count_msg_target[target].values())
                for message in count_msg_target[target].keys()
            })
            for target in count_msg_target.keys()
        })

        p_target_given_message = defaultdict(lambda: defaultdict(lambda: 1e-10), {
            message: defaultdict(lambda: 1e-10, {
                target: count_target_msg[message][target] / sum(count_target_msg[message].values())
                for target in count_target_msg[message].keys()
            })
            for message in count_target_msg.keys()
        })
        return p_target, p_message_given_target, p_target_given_message

    def _compute_complexity_infoloss_accuracy(
        self, target_message,
        p_sender_message_given_target, p_sender_target_given_message,
        p_receiver_target_given_message
    ):
        p_target = self.natural_language_profile['need_probs']
        all_target = set(x['target'] for x in target_message)
        all_message = set(x['message'] for x in target_message)
        complexity = 0  # I(W,U) = sum_u_w p(u) x p(w|u) x log2( p(u|w) / p(u) )

        for u, w in product(all_target, all_message):
            cplx = (
                p_target[u] *
                p_sender_message_given_target[u][w] *
                log2(p_sender_target_given_message[w][u] / p_target[u] + 1e-10)
            )
            complexity += cplx

        info_loss = 0  # -sum_u_w p(u) x p(w|u) x log2(p(u|w))
        for u, w in product(all_target, all_message):
            info_loss += (
                -p_target[u] *
                p_sender_message_given_target[u][w] *
                log2(p_receiver_target_given_message[w][u] + 1e-10)
            )

        acc = 0  #
        for u, w in product(all_target, all_message):
            acc += (
                p_target[u] *
                p_sender_message_given_target[u][w] *
                p_receiver_target_given_message[w][u]
            )

        return {
            'complexity': complexity,
            'information_loss': info_loss,
            'accuracy': acc
        }

    def _compute_complexity_infoloss_accuracy_natural_language(self):
        target_message = self.natural_language_profile['target_message']
        (
            p_target,
            p_message_given_target,
            p_target_given_message
        ) = self._estimate_prob_given_count(target_message)
        return self._compute_complexity_infoloss_accuracy(
            target_message,
            p_message_given_target, p_target_given_message,
            p_target_given_message
        )

    def _compute_complexity_infoloss_accuracy_emerged_language(self, counts, ego):
        target_message = [{
            'target': x['target_node'],
            'message': tuple(x['message'])
        } for x in counts if x['ego_node'] == ego]
        (
            p_target,
            p_sender_message_given_target,
            p_sender_target_given_message
        ) = self._estimate_prob_given_count(target_message)

        p_receiver_target_given_message = defaultdict(lambda: defaultdict(lambda: 1e-10))
        for x in counts:
            idx, u, w = x['target_node_idx'], x['target_node'], tuple(x['message'])
            receiver_output = x['receiver_output'][0]

            for uidx in range(len(receiver_output)):
                p_receiver_target_given_message[w][NODES[uidx]] += max(1e-10, receiver_output[uidx])

        for w in p_receiver_target_given_message.keys():
            total = sum(p_receiver_target_given_message[w].values())
            for u in p_receiver_target_given_message[w].keys():
                p_receiver_target_given_message[w][u] /= total

        return self._compute_complexity_infoloss_accuracy(
            target_message,
            p_sender_message_given_target, p_sender_target_given_message,
            p_receiver_target_given_message
        )

    def get_results(self):
        return self.results