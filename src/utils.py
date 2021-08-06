import string
import matplotlib.pyplot as plt
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

def plot_font(x, choice_characters_mask=None, mode='1_row'):
    # Está a dar mal ainda! É preciso pôr a funcionar para tensores
    # Por a mostrar letras que faltam também para 5_rows!

    if mode == '5_rows':
        _, axs = plt.subplots(6, 5, figsize=(8*5/6, 8))
        for i in range(6):
            for j in range(5):
                axs[i, j].set_axis_off()
                index = i*5+j
                if index < 26:
                    axs[i, j].spy(x[i*5+j])
                else:
                    pass
    elif mode == '1_row':
        _, axs = plt.subplots(1, len(x), figsize=(24, 24))
        for i, ax in enumerate(axs):
            ax.set_axis_off()
            ax.spy(x[i])
            if choice_characters_mask is not None and choice_characters_mask[i] == 0:
                add_subplot_border(ax, width=2, color='red')
    else:
        raise Exception(f'The mode "{mode}" doesn\'t exist!')
    plt.setp(axs, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()

def pickle_dump(object, filename, save_dir=None, full_path=False):

    if full_path:
        save_dir = ''
    else:
        # By default saves to the pickle directory
        if save_dir is None:
            save_dir = BASE_DIR + 'pickle/'
        else:
            save_dir = BASE_DIR + save_dir

    with open(save_dir + filename, 'wb') as f:
        pickle.dump(object, f)

def pickle_load(filename, save_dir=None, full_path=False):

    if full_path:
        save_dir = ''
    else:
        # By default loads from the pickle directory
        if save_dir is None:
            save_dir = BASE_DIR + 'pickle/'
        else:
            save_dir = BASE_DIR + save_dir

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
    If save_dir is None, it will default to add the directory pickle to BASE_DIR
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
            save_dir = BASE_DIR + 'checkpoints/'
        else:
            save_dir = BASE_DIR + save_dir
    
    torch.save(checkpoint, save_dir + file_name)


def load_checkpoint(file_name, save_dir=None, full_path=False):

    if full_path:
        save_dir = ''
    else:
        # By default loads from the pickle directory
        if save_dir is None:
            save_dir = BASE_DIR + 'checkpoints/'
        else:
            save_dir = BASE_DIR + save_dir
    
    full_path_to_file = save_dir + file_name

    if os.path.isfile(full_path_to_file):
        checkpoint = torch.load(full_path_to_file)
        return checkpoint
    else:
        raise Exception("This path doesn't correspond to any file!")

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
