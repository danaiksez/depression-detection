import shutil
import sklearn
import torch

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from typing import Optional

from slp.util import system
from slp.util import types

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'

class CheckpointHandler(ModelCheckpoint):
    """Augment ignite ModelCheckpoint Handler with copying the best file to a
    {filename_prefix}_{experiment_name}.best.pth.
    This helps for automatic testing etc.
    Args:
        engine (ignite.engine.Engine): The trainer engine
        to_save (dict): The objects to save
    """
    def __call__(self, engine: Engine, to_save: types.GenericDict) -> None:
        super(CheckpointHandler, self).__call__(engine, to_save)
        # Select model with best loss
        _, paths = self._saved[-1]
        for src in paths:
            splitted = src.split('_')
            fname_prefix = splitted[0]
            name = splitted[1]
            dst = f'{fname_prefix}_{name}.best.pth'
            shutil.copy(src, dst)


class EvaluationHandler(object):
    def __init__(self, pbar: Optional[ProgressBar] = None,
                 validate_every: int = 1,
                 early_stopping: Optional[EarlyStopping] = None):
        self.validate_every = validate_every
        self.print_fn = pbar.log_message if pbar is not None else print
        self.early_stopping = early_stopping

    def predict_testset(self, model, test_loader):
#        import pdb; pdb.set_trace()
        model.eval()
        model.to(DEVICE)
        y_pred = []; y_true=[]
        with torch.no_grad():
            for index, batch in enumerate(test_loader):
                x, y, l = batch
                pred = model(x, l)
                y_pred.append(pred)
                y_true.append(y)

#        yp = [y[0].max(0)[1].item() for y in y_pred]
        yp = []
        for i in range(len(y_pred)):
            for j in range(len(y_pred[0])):
                yp.append(y_pred[i][j].max(0)[1].item())

        yt = []
#        yt = [y.item() for y in y_true]
        for i in range(len(y_true)):
            for j in range(len(y_true[0])):
                yt.append(y_true[i][j].item())

        f1 = sklearn.metrics.f1_score(yp, yt, average='macro')
        uar = sklearn.metrics.recall_score(yp, yt, average='macro')

        print("F1: {}".format(f1))
        print("UAR: {}".format(uar))

        return 1

    def __call__(self, engine: Engine, model, evaluator: Engine,
                 dataloader: DataLoader, test_loader: DataLoader, validation: bool = True):
        if engine.state.epoch % self.validate_every != 0:
            return
        evaluator.run(dataloader)
        system.print_separator(n=35, print_fn=self.print_fn)
        metrics = evaluator.state.metrics
        phase = 'Validation' if validation else 'Training'
        self.print_fn('Epoch {} {} results'
                      .format(engine.state.epoch, phase))
        system.print_separator(symbol='-', n=35, print_fn=self.print_fn)
        for name, value in metrics.items():
            self.print_fn('{:<15} {:<15}'.format(name, value))

        if validation and self.early_stopping:
            loss = self.early_stopping.best_score
            patience = self.early_stopping.patience
            cntr = self.early_stopping.counter
            self.print_fn('{:<15} {:<15}'.format('best loss', -loss))
            self.print_fn('{:<15} {:<15}'.format('patience left',
                                                 patience - cntr))
            system.print_separator(n=35, print_fn=self.print_fn)

        self.predict_testset(model, test_loader)


    def attach(self, model, trainer: Engine, evaluator: Engine,
               dataloader: DataLoader, test_loader:DataLoader, validation: bool = True):
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            self, model, evaluator, dataloader, test_loader,
            validation=validation)
