from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import os

import dill
import torch
import numpy as np

from src.training.utils.metrics import get_metric_function
from src.training.utils.loss import get_loss_function
from src.training.utils.optimizer import set_optimizer
from src.training.utils.scheduler import set_scheduler
from src.training.utils.early_stopping import EarlyStopping
from src.toolkit.history import History


def get_model_trainer(trainer_name, model, data, config, device, *args, **kwargs):
    if trainer_name == 'CentralTrainer':
        trainer = CentralTrainer(model, data, config, device)
    elif trainer_name == 'HyperTrainer':
        trainer = HyperTrainer(kwargs['hypernet'], model, data, config, device)
    elif trainer_name == 'DPTrainer':
        trainer = DPTrainer(model, data, config, device)
    else:
        raise NotImplementedError(f'{trainer_name} is not implemented yet..')
    return trainer


#Basic Trainer functionalities
class Trainer(ABC):
    def __init__(self, model, data, config, device, *args, **kwargs):
        assert config is not None, 'A config munch is necessary to initialize the trainer, but config is None!'
        self.model = model
        self.trn_data, self.tst_data, self.val_data = data
        self.config = config
        self.device = device
        self.debug = config.debug

        self.use_val = self.config.data.use_val

        self.loss_fn = get_loss_function(config.training.loss, model=self.model)
        eval_ds_name = self.loss_fn.name + '_L'
        self.loss_fn.eval_ds_name = 'VAL_' + eval_ds_name if self.use_val else 'TST_' + eval_ds_name
        

        self.metrics = []
        for m in config.training.metrics:
            self.metrics.append(get_metric_function(m, num_classes = self.config.num_classes, model=self.model).to(self.device))

        self._set_optimizers()

        self.current_epoch = 0

        self.muted = False
        
    def _set_optimizers(self):
        self.optimizer = set_optimizer(self.model, self.config.training.optimizer)
        if self.config.training.lr_scheduler:
            self.scheduler = set_scheduler(self.optimizer, self.config.training.validation_frequency, self.config.training.lr_scheduler, self.use_val)
        else:
            self.scheduler = None
        if self.config.training.early_stopping:
            self.early_stopper = EarlyStopping(self.config.training.early_stopping.patience, self.config.training.early_stopping.delta, self.config.training.early_stopping.metric, self.config.training.early_stopping.use_loss, self.config.training.early_stopping.subject_to, self.use_val, self.config.training.early_stopping.verbose) 
        else: 
            self.early_stopper = None

    @abstractmethod
    def train(self, epochs=None, *args, **kwargs):
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, split='VAL', *args, **kwargs):
        pass


    def track_metrics(self, outputs, targets):
        for metric_fn in self.metrics:
            metric_fn(outputs, targets)

    def compute_metrics(self):
        metrics_dict = {}
        for metric_fn in self.metrics:
            metrics_dict[metric_fn.name] = metric_fn.compute().item()
            metric_fn.reset()
        return metrics_dict

    def update_history(self, metrics_dict, split): 
        for k, v in metrics_dict.items():
            if k in ['OverallTrainedEpochs']:
                metric_name = k
            else:
                metric_name = f'{split}_{k}'
            self.history((metric_name, self.current_epoch), v)

    def log_status(self):
        if self.muted:
            return
        format = self.loss_fn.format
        trn_loss = self.history.get(f'TRN_{self.loss_fn.name}_L')
        log_str = f'Epoch: {self.current_epoch}| LR: {self.optimizer.lr()} | '+ f'Train loss ({self.loss_fn.name}): {trn_loss:{format}}'
        if self.tst_data is not None and not self.config.eval.skip_test_data_during_training:
            try:
                tst_loss = self.history.get(f'TST_{self.loss_fn.name}_L')
                log_str += f' | Test loss ({self.loss_fn.name}): {tst_loss:{format}}'
            except: 
                print(f'No test loss available to report!')
        if self.use_val and self.val_data is not None:
            try:
                val_loss = self.history.get(f'VAL_{self.loss_fn.name}_L')
                log_str += f' | Val loss ({self.loss_fn.name}): {val_loss:{format}}'
            except: 
                print(f'No validation loss available to report!')
        print(log_str)

    def get_state_path(self, path, best=False):
        assert self.checkpoint_path or path, 'If the History object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        return statepath

    def summary(self):
        self.history.summary(mode='latest')
        key_value = f'VAL_{self.loss_fn.name}_L' if self.use_val else f'TST_{self.loss_fn.name}_L'
        self.history.summary(key_value=key_value, max_key=self.loss_fn.subject_to=='max', step_key='Epoch', mode='best')


class CentralTrainer(Trainer):
    def __init__(self, model, data, config, device, *args, **kwargs):
        super().__init__(model, data, config, device, *args, **kwargs)
        self.history = History(keys=['Metric', 'Epoch'], savefile=os.path.join(config.history_path, config.experiment_name+self.config.debug_file_suffix))
        self.checkpoint_path = os.path.join(config.checkpoint_path, config.experiment_name)

    def train(self, epochs=None, *args, **kwargs):
        if epochs is None:
            epochs = self.config.training.epochs
        assert type(epochs) == int, f'epochs need to be an integer. Got {epochs} ({type(epochs)}) instead!'
        if not self.muted:
            print(f'Start training for {epochs} epochs.')
        for epoch in range(epochs):
            self.current_epoch += 1
            self.train_step(*args, **kwargs)

            self._pre_val_scheduler_step()

            if epoch % self.config.training.validation_frequency == 0 or epoch == (epochs-1):
                self.evaluate('VAL', *args, **kwargs)
                if self.config.eval.skip_test_data_during_training:
                    pass
                else:
                    self.evaluate('TST', *args, **kwargs)

                if self.early_stopper:
                    self.early_stopper(self.history.get(self.early_stopper.metric))

                self._post_val_scheduler_step()

                self.log_status()
                self.save()

            if self.current_epoch in self.config.training.save_rounds or epoch == (epochs-1):
                self.save(self.checkpoint_path+f'_e{self.current_epoch}')
            
            if self.early_stopper:
                if self.early_stopper.improved:
                    self.save(self.checkpoint_path, best=True)
                if self.early_stopper.stop:
                    print(f'Early stopping the training since we had no improvement of {self.early_stopper.metric} for {self.early_stopper.patience} rounds. Training was stopped after {epoch} epochs')
                    break
            else:
                #if we do not use early stopping, assume that the latest model is the best model and therefore save it
                self.save(self.checkpoint_path, best=True)
            
            if self.config.debug:
                break
    
    def _pre_val_scheduler_step(self):
        if self.scheduler and not self.scheduler.after_val:
            self.scheduler.step()
            self.history(('LR', self.current_epoch), self.scheduler.lr())
    
    def _post_val_scheduler_step(self):
        if self.scheduler and self.scheduler.after_val:
            if hasattr(self.scheduler, 'metric'):
                self.scheduler.step(self.history.get(self.scheduler.metric))
            else:
                self.scheduler.step()
            self.history(('LR', self.current_epoch), self.scheduler.lr())

    def train_step(self, *args, **kwargs):
        self.model.to(self.device)
        self.model.train()
        
        losses = []
        for batch_num, (inputs, targets) in enumerate(self.trn_data):
            # zero out grads
            self._zero_grad()

            # Transfer to GPU
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            #forward pass
            outputs = self._model_forward(inputs)

            # Get loss
            loss = self._calculate_loss(outputs, targets)           
            
            if torch.isnan(loss).any():
                print('loss is nan in batch {}!'.format(batch_num))
                raise Exception('loss is nan in batch {}!'.format(batch_num))

            loss.backward()

            if self.config.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clipping)
            
            self._optimizer_step()

            #calculate all metrics
            self.track_metrics(outputs, targets)
            losses.append(loss.item())
            
            if self.config.debug:
                break

        epoch_metrics = self.compute_metrics()
        epoch_metrics[self.loss_fn.name+'_L'] = np.mean(losses)
        self.update_history(epoch_metrics, 'TRN')
        self._zero_grad()

    def _zero_grad(self):
        self.optimizer.zero_grad()
        self.model.zero_grad()

    def _model_forward(self, inputs):
        return self.model(inputs)
    
    def _calculate_loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    def _optimizer_step(self):
        self.optimizer.step()

    def evaluate(self, split='VAL', *args, **kwargs):
        if split == 'TRN':
            data = self.trn_data
        elif split == 'TST':
            data = self.tst_data
        elif split == 'VAL':
            data = self.val_data
        else:
            raise ValueError(f'Cannot interpret split "{split}"! Use one of [TRN, TST, VAL]')
        if data is None:
            print(f'Skipping evaluation for {split} split since the loader is None!')
            return
        losses = []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(data):
                # Transfer to GPU
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Get loss and metric
                outputs = self._model_forward(inputs)
                losses.append(self._calculate_loss(outputs, targets).item())
                self.track_metrics(outputs, targets)
                if self.config.debug:
                    break
        
        if 'metric_tag' in kwargs:
            split = f'{split}_{kwargs["metric_tag"]}'
        else:
            split = split
        eval_metrics = self.compute_metrics()
        eval_metrics[self.loss_fn.name+'_L'] = np.mean(losses)
        self.update_history(eval_metrics, split)

    def _model_forward(self, inputs):
        return self.model(inputs)

    def save(self, path=None, best=False):
        statepath = self.get_state_path(path, best)
        self.history.save()
        if path is None:
            self.model.save()
        else:
            if best:
                path += '_best'
            self.model.save(path)
        
        with open(statepath, 'wb') as file:
            dill.dump(self.state_dict(), file)
        #print(f'Saved trainer state to {statepath}.')
    
    def load(self, path=None, best=False):
        assert self.checkpoint_path or path, 'If the History object has no default checkpoint_path you have to provide a path!'
        path = self.checkpoint_path if path is None else path
        if best:
            path += '_best'
        statepath = path if path.endswith('.state') else path + '.state'
        with open(statepath, 'rb') as file:
            state = dill.load(file)
        self.history.load()
        self.model.load(path)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.early_stopper is not None:
            self.early_stopper.set_state(state['early_stopper'])
        self.current_epoch = state['current_epoch']

        print(f'Loaded trainer state from {statepath}.')

    def load_best(self):
        self.load(self.checkpoint_path, best=True)

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'early_stopper': self.early_stopper.get_state() if self.early_stopper is not None else None,
                'current_epoch': self.current_epoch,}
    
