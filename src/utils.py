import os
import string
import pickle
import pytz
from datetime import date, datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Auxiliary functions to deal with parameters
# !!! say where they were taken from.

def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)

def get_frozen(model_params):
    return (p for p in model_params if not p.requires_grad)

def all_trainable(model_params):
    return all(p.requires_grad for p in model_params)

def all_frozen(model_params):
    return all(not p.requires_grad for p in model_params)

def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False

def discretize(x, threshold=0.5):
    return where(x < threshold, 0, 1)

def where(cond, x, y):  # modificar para usar torch.where
    return cond*x + ~cond*y

# some plotting functions

def plot_character_old(x):
    # The function I used when the image had 0's and 255's
    # Subtrai x a 255 para ficar a letra a preto e o fundo a branco
    plt.imshow(255-x, cmap='Greys')
    plt.xlim([0, x.shape[0]])
    # Pareceu-me que tenho que pôr assim para ficar bem... :
    plt.ylim([x.shape[1], 0])
    plt.show()

def plot_character(x):
    plt.spy(x)
    plt.show()

def plot_font(x, letter_masks=None, mode='1_row', to_show=True, to_save=False, save_dir=None, filename=None):
    # Está a dar mal ainda! É preciso pôr a funcionar para tensores
    # Por a mostrar letras que faltam também para 5_rows!

    if to_save and (save_dir is None or filename is None):
        raise Exception('Introduce a valid directory and filename!')

    if mode == '4_rows':
        fig, axs = plt.subplots(4, 7, figsize=(8, 8*4/7))
        shape = x[0].shape
        index = 0
        for i in range(4):
            j = 0
            while j < 7:
                axs[i, j].set_axis_off()
                skip_plot = i != 3 or j != 0
                if skip_plot and index < 26:
                    axs[i, j].imshow(x[index], cmap='Greys', norm=plt.Normalize(0., 1.))
                    index += 1
                j += 1
        
    elif mode == '2_rows':
        fig, axs = plt.subplots(2, 13, figsize=(15, 15*2/13))
        shape = x[0].shape
        index = 0
        for i in range(2):
            for j in range(13):
                axs[i, j].set_axis_off()
                if index < 26:
                    axs[i, j].imshow(x[index], cmap='Greys', norm=plt.Normalize(0., 1.))
                index += 1

    elif mode == '1_row':
        fig, axs = plt.subplots(1, len(x), figsize=(24, 24))
        for i, ax in enumerate(axs):
            ax.set_axis_off()
            ax.imshow(x[i], cmap='Greys', norm=plt.Normalize(0., 1.))
            if letter_masks is not None and letter_masks[i] == 1:
                add_subplot_border(ax, width=2, color='red')
    else:
        raise Exception(f'The mode "{mode}" doesn\'t exist!')
    plt.setp(axs, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if to_save:
        plt.savefig(save_dir + filename + '.png', dpi=200, format='png', bbox_inches='tight')
    if to_show:
        plt.show()
    else:
        # this closes the figure, preventing it from being shown by the notebook
        # (which we might want if we want to produce a lot of pictures)
        plt.close(fig)

# taken with mods from https://stackoverflow.com/questions/45441909/how-to-add-a-fixed-width-border-to-subplot
def add_subplot_border(ax, width=0, color=None):
    fig = ax.get_figure()

    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    x0 -= width
    x1 += width
    y0 -= width
    y1 += width

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                            color=color,
                            transform=ax.transAxes,
                            zorder=-1)

    fig.patches.append(rect)
    rect.set_clip_on(False)

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.imshow(x[0], cmap='Greys')
# add_subplot_border(ax, width=2, color='red')


# autoAxis = sub1.axis()
# rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=2)
# rec = sub1.add_patch(rec)

def pickle_dump(object, filename, save_dir=None, full_path=False):

    if full_path:
        save_dir = ''
    else:
        # By default saves to the pickle directory
        if save_dir is None:
            save_dir = 'pickle/'

    with open(save_dir + filename, 'wb') as f:
        pickle.dump(object, f)

def pickle_load(filename, save_dir=None, full_path=False):

    if full_path:
        save_dir = ''
    else:
        # By default loads from the pickle directory
        if save_dir is None:
            save_dir = 'pickle/'

    with open(save_dir + filename, 'rb') as f:
        return pickle.load(f)
    
def extract_model_from_checkpoint(checkpoint, model=None, optimizer=None, 
                                  best_or_last_model='best', 
                                  load_instances_or_state='instances', 
                                  send_to_cuda=True, eval=True):
    
    # check out how to use a losslogger = checkpoint['losslogger']
    if not best_or_last_model in ['best', 'last']:
        raise Exception('You must choose either the best or the last model!')
    if not load_instances_or_state in ['instances', 'state']:
        raise Exception('You must choose to get the instances returned, or the '
                        'model and optimizer updated with the saved state!')
    
    if load_instances_or_state == 'instances':
        if best_or_last_model == 'best':
            model = checkpoint['best_model']
            optimizer = checkpoint['best_optimizer']
        else:
            model = checkpoint['last_model']
            optimizer = checkpoint['last_optimizer']
        return model, optimizer
    else:
        # if load_instances_or_state is state, just change the state of model and optimizer
        assert model is not None and optimizer is not None
        if best_or_last_model == 'best':
            model.load_state_dict(checkpoint['best_model_state_dict'])
            optimizer.load_state_dict(checkpoint['best_optimizer_state_dict'])
        else:
            model.load_state_dict(checkpoint['last_model_state_dict'])
            optimizer.load_state_dict(checkpoint['last_optimizer_state_dict'])   
    
    if send_to_cuda:
        model.to('cuda')
    if eval:
        model.eval()
    
    if load_instances_or_state == 'instances':
        return model, optimizer

def save_checkpoint(file_name, save_dir=None, best_model=None, last_model=None,
                    best_optimizer=None, last_optimizer=None, to_save='everything',
                    send_to_cpu=False, full_path=False, **kwargs):
    '''We can save the best_model, last_model, best_optimizer, last_optimizer, 
    and as kwargs the best_epoch, last_epoch, minimum_loss, batch_size, comments
    (a string with extra notes), etc.
    If save_dir is None, it will default to add the directory pickle to the base directory.
    By default, we send to CPU because it's safer this way I think
    '''

    def add_pytorch_model_or_optim_to_checkpoint(pytorch_object, object_name, to_save):
        nonlocal checkpoint
        if to_save not in ['everything', 'only_model', 'only_state_dict']:
            raise TypeError('to_save must indicate whether to save the module,'
                            ' only the state_dict, or both!')
        if send_to_cpu and isinstance(pytorch_object, torch.optim.Optimizer):
            raise TypeError("send_to_cpu doesn't apply for Optimizer objects!")

        if to_save == 'everything':
            if isinstance(pytorch_object, nn.Module) and send_to_cpu:
                checkpoint[object_name] = pytorch_object.to('cpu')
            else:
                checkpoint[object_name] = pytorch_object
            checkpoint[object_name+'_state_dict'] = pytorch_object.state_dict()
        elif to_save == 'only_model':
            if isinstance(pytorch_object, nn.Module) and send_to_cpu:
                checkpoint[object_name] = pytorch_object.to('cpu')
            else:
                checkpoint[object_name] = pytorch_object
        elif to_save == 'only_state_dict':
            checkpoint[object_name+'_state_dict'] = pytorch_object.state_dict()

    checkpoint = {}
    if best_model is not None:
        add_pytorch_model_or_optim_to_checkpoint(best_model, 'best_model', to_save)
    if last_model is not None:
        add_pytorch_model_or_optim_to_checkpoint(last_model, 'last_model', to_save)
    if best_optimizer is not None:
        add_pytorch_model_or_optim_to_checkpoint(best_optimizer, 'best_optimizer', to_save)
    if last_optimizer is not None:
        add_pytorch_model_or_optim_to_checkpoint(last_optimizer, 'last_optimizer', to_save)
    checkpoint.update(kwargs)

    if full_path:
        save_dir = ''
    else:
        # By default loads from the pickle directory
        if save_dir is None:
            save_dir = 'checkpoints/'
    
    torch.save(checkpoint, save_dir + file_name)


def load_checkpoint(file_name, save_dir=None, full_path=False):

    if full_path:
        save_dir = ''
    else:
        # By default loads from the pickle directory
        if save_dir is None:
            save_dir = 'checkpoints/'
    
    full_path_to_file = save_dir + file_name

    if os.path.isfile(full_path_to_file):
        checkpoint = torch.load(full_path_to_file)
        return checkpoint
    else:
        raise Exception("This path doesn't correspond to any file!")

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    #Make differentiable!
    def forward(self, l):
        max_abs = torch.max(torch.abs(l))
        return l/max_abs

squeeze = Squeeze()

def get_usual_energy_net_params(model):
    # TODO: Não criar lista de parametros, mas iterador
    ps = list(model.parameters())[0:36] + \
         list(model.parameters())[76:]
    return ps

def get_usual_energy_net_fc_params(model):
    # TODO: Não criar lista de parametros, mas iterador
    # It doesn't return the bias of the last linear layer in purpose
    # as this one will not not be involved in the derivatie
    ps = list(model.parameters())[76:]
    return ps

def get_usual_energy_net_named_params(model):
    # TODO: Não criar lista de parametros, mas iterador
    ps = list(energyNet.named_parameters())[0:36] + \
         list(energyNet.named_parameters())[76:]
    return ps

def get_usual_energy_net_named_fc_params(model):
    # TODO: Não criar lista de parametros, mas iterador
    ps = list(model.named_parameters())[76:]
    return ps

# def set_usual_energy_net_params(model, new_parameter_grads):
#     ps = list(model.parameters())[0:36] + list(model.parameters())[76:]
#     for i, p in enumerate(ps):
#         p.grad = new_parameter_grads[i]

eps = torch.finfo(torch.float32).eps

class AvoidZeroOne(nn.Module):
    def __init__(self):
        super(AvoidZeroOne, self).__init__()
    def forward(x):
        eps_tensor = torch.tensor(eps).expand(x.size()).to(device)
        x = torch.where(x==0., eps_tensor, x)
        x = torch.where(x==1., 1-eps_tensor, x)
        return x

def Histogram(a, bins=50, density=True, new_figure=True, figsize=(10,7), vals=False, *args, **kwargs):
    hist, bins = np.histogram(a, bins, density=density)
    bincenters = 0.5*(bins[1:] + bins[:-1])
    if new_figure:
        plt.figure(0, figsize=figsize)
    plt.plot(bincenters, hist, *args, **kwargs)
    if vals:
        return bincenters, hist

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    # This one would be the case if we'd have dimension 1 corresponding to the logits of a certain distribution 
    # def forward(self, logits):
    #     p_log_p = F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
    #     h = - p_log_p.sum(dim=1)
    #     return h

    def forward(self, logits):
        n = logits.shape[0]
        p = AvoidZeroOne.forward(torch.sigmoid(logits))
        t = p*torch.log(p) + (1.-p)*torch.log(1.-p)
        h = -t.sum((1, 2))
        return h

entropy = Entropy()

def create_hparams_dict(last_softplus, init_mode, n_iter_inference, entropy_factor, inner_lr, lr):
    hparams_dict = {'last_softplus': last_softplus,
                    'init_mode': init_mode,
                    'n_iter_inference': n_iter_inference,
                    'entropy_factor': entropy_factor,
                    'inner_lr': inner_lr,
                    'outer_lr': lr}
    return hparams_dict


def ask_add_epochs(default_n_epochs=5):
    valid_result = False
    while valid_result is not True:
        prompt = f'Press "c" to continue to the next probability setting.\n'\
                 f'You can also press Enter to add a new epoch, indicate how many epochs to add to this one, '\
                 f'or press "d" to add the default number of epochs ({default_n_epochs}).\n'
        r = input(prompt)   
        if r == 'd':
            r = str(default_n_epochs)
        if r.isnumeric():
            print(f'Add {int(r)} epoch(s)!')
            valid_result = True
            return int(r)
        elif r == '':
            print(f'Add 1 epoch!')
            return 1
        elif r == 'c':
            print('Continue!')
            return False
        else:
            print('Introduce a valid number of epochs!')

def probs_curriculum_learning(probs_choice, duration_learning):
    # duration_learning is the number of epochs for each probs_choice
    new_setting = True
    epoch = 1
    for setting_index, (prob_choice, n_epoch) in enumerate(zip(probs_choice, duration_learning)):
        j = 0
        while j < n_epoch:
            yield epoch, prob_choice, new_setting, setting_index
            new_setting = False
            if j == n_epoch - 1:
                try:
                    n_extra_epochs = ask_add_epochs()
                    if type(n_extra_epochs) is int:
                        n_epoch += n_extra_epochs
                    if n_extra_epochs == False:
                        new_setting = True
                        pass
                except EOFError:
                    new_setting = True
                    pass
            j += 1
            epoch += 1

def print_header_scores():
    print(f'\nEpoch  |  Custom BCE   BCE (train)   BCE (valid)   IOU (train)   IOU (valid)  |  k_valid     k_train    Max valid IoU   Max train IoU')

# create dataclass for train_metrics?
def print_scores(epoch, train_metrics, valid_metrics, k_valid, k_train, max_valid_iou, max_train_iou):
    print(f"{epoch:<4}   |  {train_metrics['custom_bce']:.3f}        {train_metrics['bce']:.3f}         {valid_metrics['bce']:.3f}         "
            f"{100*train_metrics['iou']:.2f}         {100*valid_metrics['iou']:.2f}        |  {k_valid:<7}     {k_train:<7}    "
            f"{100*max_valid_iou:.2f}           {100*max_train_iou:.2f}")

def current_time():
    time_zone_PT = pytz.timezone('Europe/London')
    return datetime.now(time_zone_PT)

def current_time_repr():
    return current_time().strftime("%Y-%m-%d|%Hh%M")

def get_created_class_attributes(object_):
    return [a for a in dir(object_) if a[:2] != '__']

def print_mem():
    print('memory:', torch.cuda.memory_reserved()/(1024**3))

# def _plot_font_old(x, mode='1_row'):
#     # Está a dar mal ainda! É preciso pôr a funcionar para tensores
#     if mode == '1_row':
#         f, axs = plt.subplots(1, len(x), figsize=(20, 52))
#         for i, ax in enumerate(axs):
#             ax.set_axis_off()
#             ax.spy(x[index2character[i]])
#     elif mode == '5_rows':
#         f, axs = plt.subplots(6, 5, figsize=(8*5/6, 8))
#         for i in range(6):
#             for j in range(5):
#                 axs[i, j].set_axis_off()
#                 index = i*5+j
#                 if index < 26:
#                     axs[i, j].spy(x[index2character[i*5+j]])
#                 else:
#                     pass
#     else:
#         raise Exception(f'The mode "{mode}" doesn\'t exist!')
#     plt.setp(axs, xticks=[], yticks=[])
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.show()
