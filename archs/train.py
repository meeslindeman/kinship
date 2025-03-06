import egg.core as core
import torch
from options import Options
from egg.core.callbacks import TemperatureUpdater
from analysis.logger import ResultsCollector

def perform_training(opts: Options, train_loader, val_loader, eval_loader, game):
    core.init(params=[f'--random_seed={opts.random_seed}',
                      f'--lr={opts.learning_rate}',
                      '--optimizer=adam'])

    # Initialize your custom callback with eval_loader
    callbacks = [ResultsCollector(options=opts, game=game, eval_loader=eval_loader)]

    if opts.gs_annealing:
        temp_updater = TemperatureUpdater(
            agent=game.sender,
            decay=opts.gs_tau_decay,
            minimum=opts.gs_tau_min,
            update_frequency=1 # every epoch
        )
        callbacks.append(temp_updater)

    optimizer = torch.optim.Adam(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
        device=opts.device
    )

    trainer.train(n_epochs=opts.n_epochs, opts=opts)
    core.close()

    results = callbacks.get_results()

    return results, trainer