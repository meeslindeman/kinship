import egg.core as core
import torch 
from options import Options
from analysis.logger import ResultsCollector

def perform_training(opts: Options, train_loader, val_loader, game):
    core.init(params=[f'--random_seed={opts.random_seed}',
                      '--lr=1e-3',
                      '--optimizer=adam'])
    
    callbacks = ResultsCollector(options=opts, 
                                 game=game,
                                 print_train_loss=True,
                                 compute_topsim_train_set=True,
                                 compute_topsim_test_set=True)

    optimizer = torch.optim.Adam(game.parameters())

    trainer = core.Trainer(
        game=game, 
        optimizer=optimizer, 
        train_data=train_loader,
        validation_data=val_loader, 
        callbacks=[callbacks]
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

    results = callbacks.get_results()
    
    return results, trainer
