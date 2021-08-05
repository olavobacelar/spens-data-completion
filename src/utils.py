import string
import matplotlib.pyplot as plt

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

def hamming_loss(y_pred, y):
  if y_pred.shape != y.shape:
    raise Exception('The tensors don\'t have the same shape!')
  return (y_pred != y).sum().item()/y.numel()

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
