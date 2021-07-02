# Funções auxiliares para lidar com os parametros

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

def up_down(x, threshold=0.5):
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

def plot_font(x, mode='1_row'):
    # Está a dar mal ainda! É preciso pôr a funcionar para tensores
    if mode == '1_row':
        f, axs = plt.subplots(1, len(x), figsize=(20, 52))
        for i, ax in enumerate(axs):
            ax.set_axis_off()
            ax.spy(x[index2character[i]])
    elif mode == '5_rows':
        f, axs = plt.subplots(6, 5, figsize=(8*5/6, 8))
        for i in range(6):
            for j in range(5):
                axs[i, j].set_axis_off()
                index = i*5+j
                if index < 26:
                    axs[i, j].spy(x[index2character[i*5+j]])
                else:
                    pass
    else:
        raise Exception(f'The mode "{mode}" doesn\'t exist!')
    plt.setp(axs, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# converters of indices to characters

character2index = {character: i for i, character in enumerate(string.ascii_uppercase)}
index2character = {i: character for character, i in character2index.items()}