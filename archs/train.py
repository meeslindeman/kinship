import egg.core as core
import torch
from options import Options
from analysis.logger import ResultsCollector

def perform_training(opts: Options, train_loader, val_loader, eval_loader, game):
    core.init(params=[f'--random_seed={opts.random_seed}',
                      f'--lr={opts.learning_rate}',
                      '--optimizer=adam'])

    # Initialize your custom callback with eval_loader
    callbacks = ResultsCollector(options=opts, game=game, eval_loader=eval_loader)

    optimizer = torch.optim.Adam(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=[callbacks],
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

    results = callbacks.get_results()

    return results, trainer