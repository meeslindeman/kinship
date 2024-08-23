import torch
import egg.core as core
import numpy as np
import math
from collections import Counter
from torch.distributions import Categorical

class ResultsCollector(core.Callback):
    def __init__(self, options, print_train_loss=True, compute_topsim_train_set=False, compute_topsim_test_set=True):
        self.options = options
        self.print_train_loss = print_train_loss
        self.topsim_calculator = core.TopographicSimilarity(
            sender_input_distance_fn="edit", 
            message_distance_fn="edit", 
            compute_topsim_train_set=compute_topsim_train_set, 
            compute_topsim_test_set=compute_topsim_test_set, 
            is_gumbel=True
        )
        self.results = []

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        train_metrics = self._aggregate_metrics(loss, logs, "train", epoch)

        if self.options.compute_topsim:
            topsim = self.topsim_calculator.compute_topsim(
                torch.flatten(logs.sender_input, start_dim=1), 
                logs.message.argmax(dim=-1) if self.topsim_calculator.is_gumbel else logs.message
            )
            train_metrics["topsim"] = topsim  

        if self.options.messages:
            train_metrics["messages"] = self._messages_to_indices(logs.message)

        if self.options.targets:
            train_metrics["target_node"] = [target_node for batch in logs.aux_input for target_node in batch['target_node']]

        if self.options.ego_nodes:
            train_metrics["ego_node"] = [ego_node for batch in logs.aux_input for ego_node in batch['ego_node']]

        if self.options.sequence:
            train_metrics["sequence"] = [seq.tolist() for seq in logs.sender_input]

        if self.options.complexity:
            train_metrics["complexity"] = self._calculate_complexity(logs)

        if self.options.information_loss:
            train_metrics["information_loss"] = self._calculate_information_loss(logs)

        if self.options.kl_accuracy:
            train_metrics["kl_accuracy"] = self._calc_accuracy(logs)
        
        self.results.append(train_metrics)
        self._print_to_console({k: v for k, v in train_metrics.items() if k not in ['messages', 'sequence', 'target_node', 'ego_node']})

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        test_metrics = self._aggregate_metrics(loss, logs, "test", epoch)
        if self.options.compute_topsim:
            topsim = self.topsim_calculator.compute_topsim(
                torch.flatten(logs.sender_input, start_dim=1), 
                logs.message.argmax(dim=-1) if self.topsim_calculator.is_gumbel else logs.message
            )
            test_metrics["topsim"] = topsim  

        if self.options.messages:
            test_metrics["messages"] = self._messages_to_indices(logs.message)

        if self.options.targets:
            test_metrics["target_node"] = [target_node for batch in logs.aux_input for target_node in batch['target_node']]
        
        if self.options.ego_nodes:
            test_metrics["ego_node"] = [ego_node for batch in logs.aux_input for ego_node in batch['ego_node']]

        if self.options.sequence:
            test_metrics["sequence"] = [seq.tolist() for seq in logs.sender_input]

        if self.options.complexity:
            test_metrics["complexity"] = self._calculate_complexity(logs)

        if self.options.information_loss:
            test_metrics["information_loss"] = self._calculate_information_loss(logs)
        
        if self.options.kl_accuracy:
            test_metrics["kl_accuracy"] = self._calc_accuracy(logs)
        
        self.results.append(test_metrics)
        self._print_to_console({k: v for k, v in test_metrics.items() if k not in ['messages', 'sequence', 'target_node', 'ego_node']})

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
            # For literal vector:
            # return messages_tensor.tolist()
            # For acutal symbol message:
            return [message.argmax(dim=-1).tolist() for message in messages_tensor]

    def _print_to_console(self, metrics: dict):
        output_message = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        print(output_message, flush=True)

    def _calculate_complexity(self, logs: core.Interaction):
        targets = [target_node for batch in logs.aux_input for target_node in batch['target_node']]
        
        if self.options.mode == 'gs': 
            messages = (logs.message.argmax(dim=-1)).tolist() 
        elif self.options.mode == 'rf': 
            # Messages already outputed with symbols
            messages = logs.message.tolist()

        # Convert messages to tuples for hashing
        messages = [tuple(m) for m in messages]

        # Calculate p(m) - frequency of each target
        target_freq = Counter(targets)
        p_m = {target: count / len(targets) for target, count in target_freq.items()}

        # Calculate q(w) - marginal probability of messages
        mssg_freq = Counter(messages)
        q_w = {msg: count / len(messages) for msg, count in mssg_freq.items()}

        # Calculate q(w|m) - probability distribution of messages given targets
        joint_counts = Counter(zip([tuple(m) for m in messages], targets))
        q_w_m = {k: v / target_freq[k[1]] for k, v in joint_counts.items()}

        # Calculate complexity: I(M; W)
        I_q_M_W = sum(
        p_m[m] * q_w_m[(w, m)] * math.log((q_w_m[(w, m)] + 1e-12) / q_w[w], 2)
        for (w, m) in joint_counts.keys())

        return I_q_M_W
    
    def _get_need_probs(self, targets):
        target_freq = Counter(targets)
        return {target: count / len(targets) for target, count in target_freq.items()} 

    def _calculate_information_loss(self, logs: core.Interaction):
        targets = [target_node for batch in logs.aux_input for target_node in batch['target_node']]
        need_probabilities = self._get_need_probs(targets)

        # Initialize dictionaries to store the sum of surprisals and the count of targets
        surprisal_sums = {target: 0.0 for target in need_probabilities}
        target_counts = {target: 0 for target in need_probabilities}

        log2 = torch.log(torch.tensor(2.0))

        # Iterate over each target and its corresponding log probabilities
        for i, target in enumerate(targets):
            if self.options.mode == 'gs':
                log_probs = logs.receiver_output[i].mean(dim=0)
            elif self.options.mode == 'rf':
                log_probs = logs.receiver_output[i]

            # Create a categorical distribution using the logits (log probabilities)
            dist = Categorical(logits=log_probs)

            # Calculate the surprisal of the target
            surprisal = -dist.logits[0].item() / log2.item()

            # Accumulate the surprisal sum for the current target
            surprisal_sums[target] += surprisal
            target_counts[target] += 1
        
        # Calculate the average surprisal for each target and the total information loss as a weighted sum of average surprisal
        avg_surprisal = {target: surprisal_sums[target] / target_counts[target] for target in surprisal_sums}
        #information_loss = sum(avg_surprisal[target] for target in need_probabilities) / len(need_probabilities)
        information_loss = sum(need_probabilities[target] * avg_surprisal[target] for target in need_probabilities)

        return information_loss
    
    def _calc_accuracy(self, logs: core.Interaction):
        targets = [target_node for batch in logs.aux_input for target_node in batch['target_node']]
        KL_total = 0.0
        epsilon = 1e-12
        P_i = torch.tensor([1.0 - epsilon, epsilon])

        for i, _ in enumerate(targets):
            log_probs = logs.receiver_output[i].mean(dim=0)
            dist = Categorical(logits=log_probs)
            Q_i = dist.probs + epsilon
            Q_i /= Q_i.sum()

            KL_i = torch.sum(Q_i * (torch.log(Q_i) - torch.log(P_i)))
            KL_total += KL_i.item()

        return KL_total / len(targets)

    def get_results(self):
        return self.results