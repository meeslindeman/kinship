import torch
import egg.core as core
import numpy as np
import math
from collections import Counter
from torch.distributions import Categorical

from dataclasses import dataclass
from typing import List


class ResultsCollector(core.Callback):
    @dataclass
    class EvaluationResults:
        targets: List[int]
        messages: List[int]
        receiver_output: List[int]

    def __init__(self, options, game, train_32_loader=None, val_32_loader=None, print_train_loss=True, compute_topsim_train_set=True, compute_topsim_test_set=True):
        self.options = options
        self.game = game
        self.train_32_loader = train_32_loader  
        self.val_32_loader = val_32_loader 
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

        if self.train_32_loader is not None:
            interaction_32 = self._evaluate_on_32_targets(self.train_32_loader)
            info_loss_32 = self._calculate_complexity(interaction_32)
            train_metrics['information_loss'] = info_loss_32

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
        
        self.results.append(train_metrics)
        self._print_to_console({k: v for k, v in train_metrics.items() if k not in ['messages', 'sequence', 'target_node', 'ego_node']})

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        test_metrics = self._aggregate_metrics(loss, logs, "test", epoch)

        if self.val_32_loader is not None:
            interaction_32 = self._evaluate_on_32_targets(self.val_32_loader)
            info_loss_32 = self._calculate_complexity(interaction_32)
            test_metrics['information_loss'] = info_loss_32

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

    def _evaluate_on_32_targets(self, loader):
        self.game.eval()

        with torch.no_grad():
            for batch in loader:
                _, labels, _, aux_input = batch
                outputs = self.game(None, labels, None, aux_input)
                targets = outputs[1].aux_input.target_node
                messages = outputs[1].message.argmax(dim=-1).tolist()
                receiver_output = outputs[1].receiver_output.mean(dim=1)

                print(receiver_output)

    def get_results(self):
        return self.results