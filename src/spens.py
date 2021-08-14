from typing import List, Tuple, Union
from dataclasses import dataclass, astuple
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch
import sys
from torch import nn
import torch.nn.functional as F
from dataset import FontsDataset
from model import EnergyNet

@dataclass
class UnrollConfigNew:
    '''To configure the unrolling algorithm set those parameters below.

    It's not exactly necessary, but is helpful to have the configuration
    as a dataclass (Google Colab still doesn't type check anything though.)
    '''

    n_iter_inference: int
    inner_lr: float
    init_mode: str
    to_squeeze: bool = False
    entropy_factor: float = 0.
    plot_y: bool = False
    plot_hist: bool = False
    to_print: bool = False

    def __post_init__(self):
        # With Google Colab we can't easily use type hints, so we add this
        # falta o logits_y_0
        self.inner_lr = torch.tensor(self.inner_lr, device=device)
        if not (isinstance(self.n_iter_inference, int) and isinstance(self.inner_lr, torch.Tensor) and 
            isinstance(self.init_mode, str) and isinstance(self.to_squeeze, bool) and
            isinstance(self.entropy_factor, float) and isinstance(self.plot_y, bool) 
            and isinstance(self.plot_hist, bool) and isinstance(self.to_print, bool)):
            raise TypeError('The parameters must be of the given types!')
        
        if not self.init_mode in ['random', 'zero', 'average', 'autoencoder']:
            raise Exception("init_mode must be 'random', 'zero', 'average', or'autoencoder'.")

    def __iter__(self):
        # Dataclass doesn't unpack by default
        return iter(astuple(self))
    
    def print_training_repr(self):
        print(f'Unroll Config: n_iter_inference: {self.n_iter_inference} • inner_lr: {self.inner_lr.item()} • '
              f'init_mode: {self.init_mode}')

  
class UnrollEnergyNew(nn.Module):
    '''
    [...] This module is to be optimized in the space of parameters of the energy
    function and the space of hyper-parameters of the optimizer (...)
    '''
    def __init__(self, energy_net, loss_function, unroll_config):
        # If I pass energy_net here pytorch automatically assumes that I want
        # the parameters of energy_net to be parameters of the module
        # But i have extra (decoding parameters) that I don't want to include.
        # Is it even a problem?

        '''[...]'''
        super(UnrollEnergyNew, self).__init__()
        self.energy_net = energy_net
        self.loss_function = loss_function
        self.n_iter_inference, self.inner_lr, self.init_mode, self.to_squeeze, \
        self.entropy_factor, self.plot_y, self.plot_hist, self.to_print = unroll_config
        
    def forward(self, x, y, letter_masks, lr, create_graph=True, logits_y_0=None):
        # The ground truth y is here for the computation of the loss
        
        current_batch_size = x.shape[0] 
        shape_logits_0 = ((~letter_masks).sum().item(), *LETTER_SHAPE)
        
        if logits_y_0 is None:
            if self.init_mode == 'random':
                logits_y = torch.randn(shape_logits_0).to(device)
            elif self.init_mode == 'zero':
                logits_y = torch.zeros(shape_logits_0).to(device)
            elif self.init_mode == 'average':
                logits_y = average_of_letter_for_init[:current_batch_size][~letter_masks].to(device)
            elif self.init_mode == 'autoencoder':
                logits_y = autoencoder(x)[~letter_masks].to(device)

            if self.to_squeeze: # compress the logits with the custom squeeze function 
                logits_y = squeeze(logits_y)
        else:
            assert logits_y_0.shape == shape_logits_0
            logits_y = logits_y_0

        loss = torch.zeros(1, requires_grad=True, device=device)

        with torch.no_grad():
            y_ = torch.sigmoid(logits_y)
            y_pred_discrete = discretize(y_)

        if self.plot_hist:
            plt.hist(logits_y.detach().cpu().numpy().flatten(), 100); plt.show()
            # plt.hist(y_.detach().cpu().numpy().flatten(), 100); plt.show()

        if self.plot_y:
            plt.imshow(y_pred_discrete[0].cpu().detach().numpy()); plt.show()

        for n in range(self.n_iter_inference):
            
            logits_y = logits_y.clone().detach().requires_grad_()
            x[~letter_masks] = torch.sigmoid(logits_y)
            
            # Não devia ser negative entropy?
            if self.entropy_factor == 0.:
                energy = self.energy_net(x.clone(), mode='usual')
            else:
                energy = self.energy_net(x.clone(), mode='usual') + self.entropy_factor*entropy(logits_y)
            
            # Compute the gradient of the energy wrt y_.
            # This gradient grad_E_y is itself differentiable since create_graph=True
            grad_E_logits_y, = torch.autograd.grad(energy, inputs=logits_y,
                                        grad_outputs=torch.ones_like(energy),
                                        create_graph=create_graph)

            logits_y = logits_y - lr*grad_E_logits_y
            loss = loss + self.loss_function(logits_y, y)/(self.n_iter_inference - n)

            with torch.no_grad():
                y_ = torch.sigmoid(logits_y)
                y_pred_discrete = discretize(y_)



            if self.to_print:
                print('logits min max', logits_y.min().item(), logits_y.max().item())
                print('grad_E_logits_y min max', grad_E_logits_y.min().item(), grad_E_logits_y.max().item())

            if self.plot_y:
                plt.imshow(y_pred_discrete[0].cpu().detach().numpy()); plt.show()

            if self.plot_hist:
                plt.hist(logits_y.detach().cpu().numpy().flatten(), 100); plt.show()

        loss = loss/self.n_iter_inference
        return logits_y, loss

@dataclass
class UnrollConfig:
    '''To configure the unrolling algorithm set those parameters below.

    It's not exactly necessary, but is helpful to have the configuration
    as a dataclass (Google Colab still doesn't type check anything though.)
    '''

    n_iter_inference: int
    inner_lr: float
    random_init: bool = False
    init_mode: str = 'pass'
    average_of_letter_init: bool = False
    entropy_factor: float = 0.
    plot_y: bool = False
    plot_hist: bool = False
    to_print: bool = False
    baseline_init: bool = False

    def __post_init__(self):
        # With Google Colab we can't easily use type hints, so we add this
        # falta o logits_y_0
        self.inner_lr = torch.tensor(self.inner_lr, device=device)
        if not (isinstance(self.n_iter_inference, int) and isinstance(self.inner_lr, torch.Tensor) and 
            isinstance(self.random_init, bool) and isinstance(self.init_mode, str) and isinstance(self.average_of_letter_init, bool) and
            isinstance(self.entropy_factor, float) and isinstance(self.plot_y, bool) 
            and isinstance(self.plot_hist, bool) and isinstance(self.to_print, bool)):
            raise TypeError('The parameters must be of the given types!')
        
        if not self.init_mode in ['pass', 'squeeze']:
            raise Exception("init_mode must be either 'pass' (do nothing) or 'squeeze'")

    def __iter__(self):
        # Dataclass doesn't unpack by default
        return iter(astuple(self))
    
    def print_training_repr(self):
        print(f'Unroll Config: n_iter_inference: {self.n_iter_inference} • inner_lr: {self.inner_lr.item()} • '
              f'random_init: {self.random_init} • entropy_factor: {self.entropy_factor}')

class UnrollEnergy(nn.Module):
    '''
    [...] This module is to be optimized in the space of parameters of the energy
    function and the space of hyper-parameters of the optimizer (...)
    '''
    def __init__(self, energy_net, loss_function, unroll_config):
        # If I pass energy_net here pytorch automatically assumes that I want
        # the parameters of energy_net to be parameters of the module
        # But i have extra (decoding parameters) that I don't want to include.
        # Is it even a problem?

        '''[...]'''
        super(UnrollEnergy, self).__init__()
        self.energy_net = energy_net
        self.loss_function = loss_function
        self.n_iter_inference, self.inner_lr, self.random_init, self.init_mode, self.average_of_letter_init, \
        self.entropy_factor, self.plot_y, self.plot_hist, self.to_print = unroll_config
        
    def forward(self, x, y, letter_masks, lr, create_graph=True, logits_y_0=None):
        # The ground truth y is here for the computation of the loss
        
        current_batch_size = x.shape[0] 
        shape_logits_0 = ((~letter_masks).sum().item(), *LETTER_SHAPE)
        
        if logits_y_0 is None:
            if self.random_init:
                logits_y = torch.randn(shape_logits_0).to(device)
            elif self.average_of_letter_init:
                logits_y = average_of_letter_for_init[:current_batch_size][~letter_masks].to(device)

            else:
                logits_y = torch.zeros(shape_logits_0).to(device)

            if self.init_mode == 'pass': # usual mode without change of the logits
                pass
            elif self.init_mode == 'squeeze':  # compress the logits with the custom squeeze function
                logits_y = squeeze(logits_y)
        else:
            assert logits_y_0.shape == shape_logits_0
            logits_y = logits_y_0

        loss = torch.zeros(1, requires_grad=True, device=device)

        with torch.no_grad():
            y_ = torch.sigmoid(logits_y)
            y_pred_discrete = discretize(y_)

        if self.plot_hist:
            plt.hist(logits_y.detach().cpu().numpy().flatten(), 100); plt.show()
            # plt.hist(y_.detach().cpu().numpy().flatten(), 100); plt.show()

        if self.plot_y:
            plt.imshow(y_pred_discrete[0].cpu().detach().numpy()); plt.show()

        for n in range(self.n_iter_inference):
            
            logits_y = logits_y.clone().detach().requires_grad_()
            x[~letter_masks] = torch.sigmoid(logits_y)

            # Não devia ser negative entropy?
            if self.entropy_factor == 0.:
                energy = self.energy_net(x.clone(), mode='usual')
            else:
                energy = self.energy_net(x.clone(), mode='usual') + self.entropy_factor*entropy(logits_y)
            
            # Compute the gradient of the energy wrt y_.
            # This gradient grad_E_y is itself differentiable since create_graph=True
            grad_E_logits_y, = torch.autograd.grad(energy, inputs=logits_y,
                                        grad_outputs=torch.ones_like(energy),
                                        create_graph=create_graph)

            logits_y = logits_y - lr*grad_E_logits_y
            loss = loss + self.loss_function(logits_y, y)/(self.n_iter_inference - n)

            with torch.no_grad():
                y_ = torch.sigmoid(logits_y)
                y_pred_discrete = discretize(y_)

            if self.to_print:
                print('logits min max', logits_y.min().item(), logits_y.max().item())
                print('grad_E_logits_y min max', grad_E_logits_y.min().item(), grad_E_logits_y.max().item())

            if self.plot_y:
                plt.imshow(y_pred_discrete[0].cpu().detach().numpy()); plt.show()

            if self.plot_hist:
                plt.hist(logits_y.detach().cpu().numpy().flatten(), 100); plt.show()
        
        x[~letter_masks] = torch.sigmoid(logits_y)

        loss = loss/self.n_iter_inference
        return logits_y, loss

@dataclass(frozen=True)
class TrainingConfig:
    '''To configure the training routine set those parameters below.

    It's not exactly necessary, but is helpful to have the configuration
    as a dataclass (Google Colab still doesn't type check anything though.)
    '''

    outer_lr: float
    n_epochs: int
    k_patience: int
    inner_lr_is_parameter: bool = False
    use_new_unroll_config: bool = True
    saved_checkpoint: Union[dict, None] = None
    metrics_names: Tuple = ('bce', 'custom_bce', 'iou')
    force_init_zero_validation: bool = True
    verbose: bool = False

    def __post_init__(self):
        # With Google Colab we can't easily use type hints, so we add this
        if not (isinstance(self.outer_lr, float) and isinstance(self.n_epochs, int) and 
            isinstance(self.k_patience, int) and isinstance(self.metrics_names, tuple) and 
            isinstance(self.force_init_zero_validation, bool) and isinstance(self.verbose, bool)):
            raise TypeError('The parameters must be of the given types!')

    def __iter__(self):
        # Dataclass doesn't unpack by default
        return iter(astuple(self))
    
    def print_training_repr(self):
        print(f'Train Config: outer_lr: {self.outer_lr}')

# class TrainEnergyNet:

# make sure the seeding is done each time the function runs!

def train_unrolled(dataset_config, energy_net_config, unroll_config, training_config):

    def make_checkpoint_dict():
        scores = {'train_iou': train_iou[:epoch], 'valid_iou': valid_iou[:epoch],
                  'train_bce': train_bce[:epoch], 'valid_bce': valid_bce[:epoch]}
        checkpoint = {'file_name': beginning_time_repr + '.dat',
                      'save_dir': 'run_logs/',
                      'duration': current_time() - beginning_time,
                      'model': unrolled,
                      'best_optimizer_state': best_optimizer_state,
                      'last_optimizer_state': last_optimizer_state,
                      'continuation': continuation,
                      'best_states': unrolled_best_states,
                      'last_states': unrolled_last_states,
                      'inner_lr': inner_lr_,
                      'best_rngs_states': best_rngs_states, 
                      'last_rngs_states': last_rngs_states,
                      'k_valid': k_valid,
                      'k_train': k_train,
                      'scores': scores,
                      'max_valid_iou': max_valid_iou,
                      'max_train_iou': max_train_iou,
                      'min_valid_bce_loss': min_valid_bce_loss,
                      'epoch': epoch,
                      'epoch_max_valid_iou': epoch_max_valid_iou,
                      'epoch_max_train_iou': epoch_max_train_iou,
                      'dataset_config': dataset_config,
                      'energy_net_config': energy_net_config,
                      'unroll_config': unroll_config,
                      'training_config': training_config}
        return checkpoint

    try:
        # Initialize dataset. Using the seed, we make sure that ... (complete)
        train_dataset = FontsDataset(train_data, dataset_config)
        valid_dataset = FontsDataset(valid_data, dataset_config)

        batch_size = dataset_config.batch_size
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False,
                                      num_workers=0, pin_memory=True)

        inner_lr = unroll_config.inner_lr

        outer_lr, n_epochs, k_patience, inner_lr_is_parameter, use_new_unroll_config, saved_checkpoint, metrics_names, \
                            force_init_zero_validation, verbose = training_config
                
        criterion = nn.BCEWithLogitsLoss().to(device)
        
        energy_net = EnergyNet(energy_net_config).to(device)
        
        if inner_lr_is_parameter:
            log_inner_lr = torch.nn.Parameter(inner_lr.log()).to(device)
            inner_lr_ = log_inner_lr.exp()
            parameters_to_optimize = list(energy_net.parameters()) + [log_inner_lr]
            
        else:
            inner_lr_ = inner_lr.to(device)
            parameters_to_optimize = list(energy_net.parameters())

        if unroll_config.init_mode == 'autoencoder':
            parameters_to_optimize = parameters_to_optimize + list(autoencoder.parameters())

        optimizer = optim.AdamW(parameters_to_optimize, lr=outer_lr, weight_decay=0.1)
        
        if use_new_unroll_config:
            unrolled = UnrollEnergyNew(energy_net, criterion, unroll_config).to(device)
            if unroll_config.init_mode == 'random' and force_init_zero_validation:
                unroll_config_valid = replace(unroll_config, init_mode='zero')
            else:
                unroll_config_valid = unroll_config
            unrolled_valid = UnrollEnergyNew(energy_net, criterion, unroll_config_valid).to(device)
        else:
            unrolled = UnrollEnergy(energy_net, criterion, unroll_config).to(device)
            if unroll_config.random_init and force_init_zero_validation:
                unroll_config_valid = replace(unroll_config, random_init=False)
            else:
                unroll_config_valid = unroll_config
            unrolled_valid = UnrollEnergy(energy_net, criterion, unroll_config_valid).to(device)

        # Print configurations and header for the scores information
        dataset_config.print_training_repr()
        energy_net_config.print_training_repr()
        unroll_config.print_training_repr()
        training_config.print_training_repr()
        print_header_scores()

        beginning_time = current_time()
        beginning_time_repr = current_time_repr()

        # Save last 3 best models, and 1 model every 100 iterations for the maximum of 3 of them
        unrolled_best_states = deque(maxlen=3)
        unrolled_last_states = deque(maxlen=3)

        # 
        train_iou = np.zeros(n_epochs)
        valid_iou = np.zeros(n_epochs)
        train_bce = np.zeros(n_epochs)
        valid_bce = np.zeros(n_epochs)

        if saved_checkpoint is None:
            continuation = True
            # Set seed to assure reproducibility of the order of the presented fonts:
            torch.manual_seed(dataset_config.initial_seed)
            min_valid_bce_loss = np.inf
            max_valid_iou = 0.
            max_train_iou = 0.
            k_valid = 0
            k_train = 0
        else:
            continuation = False
            if saved_checkpoint['epoch_max_valid_iou'] == saved_checkpoint['epoch']:
                unrolled.load_state_dict(saved_checkpoint['best_states'][-1])
                optimizer.load_state_dict(saved_checkpoint['best_optimizer_state'])
                rng_state, np_rng_state, torch_rng_state = saved_checkpoint['best_rngs_states']
            else:
                unrolled.load_state_dict(saved_checkpoint['last_states'][-1])
                optimizer.load_state_dict(saved_checkpoint['last_optimizer_state'])
                rng_state, np_rng_state, torch_rng_state = saved_checkpoint['last_rngs_states']
        
            torch.set_rng_state(torch_rng_state)
            train_dataset.set_rngs_states((rng_state, np_rng_state))

            min_valid_bce_loss = saved_checkpoint['min_valid_bce_loss']
            max_valid_iou = saved_checkpoint['max_valid_iou']
            max_train_iou = saved_checkpoint['max_train_iou']
            k_valid = saved_checkpoint['k_valid']
            k_train = saved_checkpoint['k_train']
        epoch = 1
        epoch_max_valid_iou = 1
        epoch_max_train_iou = 1
        best_optimizer_state = None
        last_optimizer_state = None
        best_rngs_states = None
        last_rngs_states = None

        while k_valid != k_patience and epoch <= n_epochs:

            if verbose:
                print(f'----- Epoch: {epoch} -----')
            # print(f'inner lr: {inner_lr_.item():.3f}')

            # Mudar o que se segue se o modo não for "prob"!

            train_dataset.build_masked_input(dataset_config)

            # if inner_lr_is_parameter:
            #     print('ho')
            #     unroll_config = replace(unroll_config, inner_lr=inner_lr_)
            #     unroll_config_valid = replace(unroll_config_valid, inner_lr=inner_lr_)
            #     unrolled = UnrollEnergy(energy_net, criterion, unroll_config).to(device)
            #     unrolled_valid = UnrollEnergy(energy_net, criterion, unroll_config_valid).to(device)

            ### TRAINING
            if verbose:
                print('--- Train ---')

            energy_net.train()

            running_metrics = init_running_metrics(metrics_names)
            total_n_missing_letters = 0
         
            for i, (x, y, letter_masks) in enumerate(train_dataloader):
                if verbose:
                    print(f'minibatch: {i}')

                # if i == 0:
                #     for j in range(3):
                #         plot_font(x[j].cpu())

                x, y, letter_masks = x.to(device), y.to(device), letter_masks.to(device)
                energy_net.zero_grad()
               
                # É preciso fazer clone().detach() aqui?
                logits_y, custom_bce_loss = unrolled(x.clone().detach(), y[~letter_masks],
                                                     letter_masks, inner_lr_)
                
                custom_bce_loss.backward()
                assert None not in get_usual_energy_net_params(energy_net) # preciso?
                optimizer.step()

                if inner_lr_is_parameter:
                    inner_lr_ = log_inner_lr.exp()

                with torch.no_grad():
                    y_ = torch.sigmoid(logits_y)
                    y_pred_discrete = discretize(y_)
                    
                running_metrics, total_n_missing_letters = update_running_metrics(running_metrics, logits_y, 
                                                                        y[~letter_masks], letter_masks, 
                                                                        total_n_missing_letters,
                                                                        custom_loss=custom_bce_loss,
                                                                        criterion=criterion)
                
            train_metrics = compute_final_metrics(running_metrics,
                                                len(train_dataloader.dataset),
                                                total_n_missing_letters)


            ### VALIDATION - Notice that Autograd is turned on here!
            if verbose:
                print('--- Valid ---')

            energy_net.eval()

            running_metrics = init_running_metrics(metrics_names)
            total_n_missing_letters = 0
            
            # tenho que passar parametro para create_graph para minimizar tempo de inferencia
            
            for i, (x, y, letter_masks) in enumerate(valid_dataloader):

                if verbose:
                    print(f'minibatch: {i}')

                x, y, letter_masks = x.to(device), y.to(device), letter_masks.to(device)
                energy_net.zero_grad()

                logits_y, custom_bce_loss = unrolled_valid(x, y[~letter_masks], letter_masks,
                                                           inner_lr_, create_graph=False)

                with torch.no_grad():
                    y_ = torch.sigmoid(logits_y)
                    y_pred_discrete = discretize(y_)

                running_metrics, total_n_missing_letters = update_running_metrics(running_metrics, logits_y, 
                                                                        y[~letter_masks], letter_masks, 
                                                                        total_n_missing_letters,
                                                                        custom_loss=custom_bce_loss, 
                                                                        criterion=criterion)

            valid_metrics = compute_final_metrics(running_metrics,
                                                len(valid_dataloader.dataset),
                                                total_n_missing_letters)
            
            train_iou[epoch-1] = train_metrics['iou']
            valid_iou[epoch-1] = valid_metrics['iou']
            train_bce[epoch-1] = train_metrics['bce']
            valid_bce[epoch-1] = valid_metrics['bce']


            ### CHECKPOINTS:
            if valid_metrics['iou'] > max_valid_iou:
                max_valid_iou = valid_metrics['iou']
                epoch_max_valid_iou = epoch
                unrolled_best_states.append(copy.deepcopy(unrolled.state_dict()))
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_rngs_states = copy.deepcopy((*train_dataset.get_rngs_states(), \
                                                             torch.get_rng_state()))
                checkpoint = make_checkpoint_dict()
                save_checkpoint(**checkpoint)
                k_valid = 0
            else:
                k_valid += 1
            
            if train_metrics['iou'] > max_train_iou:
                max_train_iou = train_metrics['iou']
                epoch_max_train_iou = epoch
                k_train = 0
            else:
                k_train += 1
            
            if epoch % 100 == 0:
                unrolled_last_states.append(copy.deepcopy(unrolled.state_dict()))
                last_optimizer_state = copy.deepcopy(optimizer.state_dict())
                last_rngs_states = copy.deepcopy((*train_dataset.get_rngs_states(), \
                                                             torch.get_rng_state()))
                checkpoint = make_checkpoint_dict()
                save_checkpoint(**checkpoint)

            # This next one is computed only for curiosity
            if valid_metrics['bce'] < min_valid_bce_loss:
                min_valid_bce_loss = valid_metrics['bce']

            print_scores(epoch, train_metrics, valid_metrics, k_valid, k_train, max_valid_iou, max_train_iou)

            epoch += 1

    except KeyboardInterrupt:
        print('\nDone! (Interrupted)\n')
        
    except:
        print(f"\nUnexpected error: {sys.exc_info()[0]}\n")
        raise

    else:        
        print('\nDone! (End of patience or maximum number of epochs exceeded!)\n')

    return unrolled_best_states, unrolled_last_states, checkpoint

@dataclass(frozen=True)
class TestingConfig:
    '''To configure the testing routine set those parameters below.

    It's not exactly necessary, but is helpful to have the configuration
    as a dataclass (Google Colab still doesn't type check anything though.)
    '''

    saved_model_parameters: Union[OrderedDict, None] = None
    saved_model: Union[nn.Module, None] = None
    n_trials: int = 100
    set_eval: bool = True
    metrics_names: Tuple = ('bce', 'custom_bce', 'iou')
    force_init_zero_testing: bool = True
    print_testing_repr: bool = False
    print_results: bool = True
    plot_fonts: bool = False
    plot_whole_font: bool = True
    plot_grid_missing_letters: bool = False

    def __post_init__(self):

        if not (self.saved_model_parameters is None) ^ (self.saved_model is None):
            raise ValueError('Either saved_model_parameters or saved_model must be chosen!')
        # With Google Colab we can't easily use type hints, so we add this
        if not (isinstance(self.n_trials, int) and isinstance(self.set_eval, bool) and 
            isinstance(self.metrics_names, tuple) and isinstance(self.print_testing_repr, bool) and 
            isinstance(self.print_results, bool) and isinstance(self.plot_fonts, bool) and
            isinstance(self.plot_whole_font, bool) and isinstance(self.plot_grid_missing_letters, bool)):
            raise TypeError('The parameters must be of the given types!')

    def __iter__(self):
        # Dataclass doesn't unpack by default
        return iter(astuple(self))
    
    def print_training_repr(self):
        print(f'Train Config: outer_lr: {self.outer_lr}')

def test_unrolled(testing_config, checkpoint=None, dataset_config=None, energy_net_config=None, unroll_config=None, dataset_to_test='test', is_unroll_new=True):

    all_configs_except_testing = (dataset_config, energy_net_config, unroll_config)
    all_configs_except_testing_are_none = all(cfg is None for cfg in all_configs_except_testing)

    if not (checkpoint is None) ^ all_configs_except_testing_are_none:
        raise Exception('Either a checkpoint must be given, or all configurations '
                        '(dataset_config, energy_net_config, and unroll_config) must be given.')
    
    if checkpoint is not None:
        dataset_config = checkpoint['dataset_config']
        energy_net_config = checkpoint['energy_net_config']
        unroll_config = checkpoint['unroll_config']
        if 'inner_lr_learned' in checkpoint:
            inner_lr = checkpoint['inner_lr_learned']
        else:
            inner_lr = unroll_config.inner_lr
    
    assert dataset_to_test in ['test', 'valid', 'train']
    
    if dataset_to_test == 'test':
        dataset = FontsDataset(test_data, dataset_config)
    elif dataset_to_test == 'valid':
        dataset = FontsDataset(valid_data, dataset_config)
    elif dataset_to_test == 'train':
        dataset = FontsDataset(train_data, dataset_config)

    # Set seed to assure reproducibility of the order of the presented fonts:
    torch.manual_seed(dataset_config.initial_seed)

    batch_size = dataset_config.batch_size
    test_dataloader = DataLoader(dataset, batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
    
    saved_model_parameters, saved_model, n_trials, set_to_eval, metrics_names, force_init_zero_testing, \
    print_testing_repr, print_results, plot_fonts, plot_whole_font, plot_grid_missing_letters = testing_config

    criterion = nn.BCEWithLogitsLoss().to(device)

    assert (saved_model_parameters is None) ^ (saved_model is None)

    if saved_model_parameters is not None:
        # por como estava: if unroll_config.random_init and force_init_zero_testing:
        if unroll_config.init_mode=='random' and force_init_zero_testing:
            unroll_config_test = replace(unroll_config, random_init=False)
        else:
            unroll_config_test = unroll_config

        energy_net = EnergyNet(energy_net_config).to(device)
        if is_unroll_new:
            unrolled_test = UnrollEnergyNew(energy_net, criterion, unroll_config_test).to(device)
        else:
            unrolled_test = UnrollEnergy(energy_net, criterion, unroll_config_test).to(device)
        unrolled_test.load_state_dict(saved_model_parameters)
    elif saved_model is not None:
        assert not unroll_config.random_init and not force_init_zero_testing, 'Not coded the possibility of having a saved model and forcing the init to be zero'
        unrolled_test = saved_model.to(device)
        energy_net = unrolled_test.energy_net
    
    if set_to_eval:
        energy_net.eval()
    else:
        energy_net.train()

    # Print configurations
    if print_testing_repr:
        dataset_config.print_training_repr()
        energy_net_config.print_training_repr()
        unroll_config.print_training_repr()
        testing_config.print_training_repr()
    
    ious = np.zeros(n_trials)

    for trial in range(n_trials):

        running_metrics = init_running_metrics(metrics_names)
        total_n_missing_letters = 0

        for j, (x, y, letter_masks) in enumerate(test_dataloader):
            batch_size_ = x.shape[0]
            x, y, letter_masks = x.to(device), y.to(device), letter_masks.to(device)
            energy_net.zero_grad()

            logits_y, custom_bce_loss = unrolled_test(x, y[~letter_masks], letter_masks,
                                                    inner_lr, create_graph=False)
            loss_usual = criterion(logits_y, y[~letter_masks])

            with torch.no_grad():
                y_ = torch.sigmoid(logits_y).cpu().detach().numpy()
                y_pred_discrete = discretize(y_)

            running_metrics, total_n_missing_letters = update_running_metrics(running_metrics, logits_y, 
                                                                    y[~letter_masks], letter_masks, 
                                                                    total_n_missing_letters,
                                                                    custom_loss=custom_bce_loss,
                                                                    criterion=criterion)
            if plot_fonts:
                # retirar a parte de gravar em disco!
                for i in range(batch_size_):
                    x_ = copy.deepcopy(x.detach())
                    x_[~letter_masks] = torch.tensor(y_pred_discrete, dtype=torch.float32).to(device)   # por mais simples
                    # plot_font(x_[i].cpu().detach().numpy(), letter_masks[i].cpu().detach().numpy(), mode='2_rows', \
                    #           to_show=False, to_save=True, save_dir='images_for_plots/spens/' + str(dataset_config.n_visible_letters)+'/', \
                    #           filename=f'n_vis={dataset_config.n_visible_letters} - {j*batch_size + i} - results')
                    # plot_font(x[i].cpu().detach().numpy(), mode='2_rows', \
                    #           to_show=False, to_save=True, save_dir='images_for_plots/spens/' + str(dataset_config.n_visible_letters)+'/', \
                    #           filename=f'n_vis={dataset_config.n_visible_letters} - {j*batch_size + i} - masked') # retirar este na versão final!
                    if plot_whole_font:
                        plot_font(y[i].cpu().detach().numpy(), mode='2_rows', \
                              to_show=False, to_save=True, save_dir='images_for_plots/spens/' + str(dataset_config.n_visible_letters)+'/', \
                              filename=f'n_vis={dataset_config.n_visible_letters} - {j*batch_size + i} - complete')

            if plot_grid_missing_letters:
                plt.figure(0, figsize=(25, 25))
                img = torchvision.utils.make_grid(y_pred_discrete.unsqueeze(1), nrow=25)[0]
                plt.imshow(img.cpu().detach())
                plt.show()

        test_metrics = compute_final_metrics(running_metrics,
                                            len(dataset),
                                            total_n_missing_letters)

        if print_results:
            print(f"BCE: {test_metrics['bce']:.3f} | IoU: {100*test_metrics['iou']:.2f} %")
        ious[trial] = test_metrics['iou']
        dataset.build_masked_input(dataset_config)

    mean_ious = 100*np.mean(ious)
    std_ious = 100*np.std(ious)

    print(f'IoU - Mean: {mean_ious:.2f} • Standard Deviation: {std_ious:.2f}')

    return ious, mean_ious, std_ious
