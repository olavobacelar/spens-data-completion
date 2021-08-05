# Preparar os dados no __init__ para ser mais fácil depois.
#

import pandas as pd
import numpy as np
import numpy.random as rd
import torch
from torch.utils.data import Dataset, DataLoader

N_CHARACTERS = 26  # tirar daqui, precisa de estar também no modulo que chama. E não devia estar hardcoded

class FontsDataset(Dataset):
    # É ineficiente transformar em listas??? Tentar mudar para tensores mesmo?
    def __init__(self, data, prob_choice=0.8, seed=None, has_font_masks=False):
        # y is the (unmasked) output. We'll need to construct the masked input
        # # Não uso o mode, que passava antes. Apagar?
        # if mode not in ['training', 'validation', 'test']:
        #     raise Exception('Invalid mode!')
        # self.mode = mode
        self.prob_choice = prob_choice

        # Data input and self.font_masks are is expected to be a pandas dataframe
        self.y = data.y.values.tolist()
        if 'font_masks' in data.columns:
            self.font_masks = data.font_masks.values.tolist()
        else:
            self.font_masks = None

        # Cuidado com as masks, pode ser lista de Nones?
        assert self.y[0].shape[0] == N_CHARACTERS
        self.n_fonts = len(self.y)
        
        self.build_masked_input(seed)
        # enviar para o gpu?
        
    def __len__(self):
        return self.n_fonts
  
    def __getitem__(self, index):
        # É mais rápido transformar em tensor e float aqui ou dentro da rotina das features?
        return torch.tensor(self.x[index]), torch.tensor(self.y[index]), self.font_masks[index]

    @staticmethod
    def new_font_masks(n_fonts, prob_choice=0.8, seed=None):
        # prob_choice is the probability of a certain character NOT being masked
        # a seed vai persistir se chamar a função mais tarde sem a escolha da seed
        # depois de já a ter usado? Preciso de pôr seed = None?
        if seed is not None:
            rd.seed(seed)
        font_masks = rd.choice((True, False), size=(n_fonts, N_CHARACTERS), p=[prob_choice, 1-prob_choice])
        font_masks = list(font_masks.astype(bool))
        return font_masks
        # Enviar para o gpu? 

    def build_masked_input(self, prob_choice=None, seed=None, font_masks=None, generate_new_masks=True):
        # if self.mode != 'training':
        #     warnings.warn('One should only call new_masked_input on initialization'
        #                  'and after each epoch in the training dataset only!')
        # I may not need to pass font_masks as I'm using the seed to try to always get the same values
        # I may not even need the seed if I just make one copy of the valid and test dataset
        # and work with those with all the data. May still be useful to have those though.
        # I'm assuming font_masks is a list. Here we use the passed font masks or create new ones
		
        if prob_choice is not None:
            self.prob_choice = prob_choice
		
        if generate_new_masks is True or font_masks is None:
            self.font_masks = FontsDataset.new_font_masks(self.n_fonts, self.prob_choice, seed)
        elif len(self.font_masks) != self.n_fonts:
            raise Exception('The number of fonts and font masks is not the same!')
        else:
            self.font_masks = font_masks

        self.x = [None]*self.n_fonts
        for font_index in range(self.n_fonts):
            self.x[font_index] = np.zeros((N_CHARACTERS, 48, 48), dtype=np.float32)
            for char_index in range(N_CHARACTERS):
                if self.font_masks[font_index][char_index] == True:
                    self.x[font_index][char_index] = self.y[font_index][char_index]
    
    def get_font_masks(self):
        # I may not need this one.
        if self.font_masks is not None:
            return self.font_masks
        else:
            raise Exception("No font masks were created yet!")


# # x = pickle.load(open('/content/drive/My Drive/thesis/data/X48.p', "rb"))
# # y = pickle.load(open('/content/drive/My Drive/thesis/data/y48.p', "rb"))

# fonts48_df = pd.read_pickle('./data/fonts48_df.pkl');
# n_fonts = fonts48_df.shape[0]

# n_characters = 26
# # n_characters_missing = 4
# # n_characters_present = n_characters - n_characters_missing

# # index_characters_missing = random.sample(range(26), n_characters_missing)
# # index_characters_missing.sort()

# # characters_missing = [index2character[i] for i in index_characters_missing]

# # characters_present = []
# # for char in string.ascii_uppercase:
# #     if char not in characters_missing:
# #         characters_present.append(char)




# fonts48_df = pd.read_pickle('./data/fonts48_df.pkl');
# fonts48_df.drop(['font_name'], axis=1, inplace=True)

# n_fonts = fonts48_df.shape[0]
# N_CHARACTERS = 26

# assert fonts48_df.shape[1] == N_CHARACTERS

# # Pôr mais bonito, sem loops (Tive que criar lista para ser mais facil criar o dataframe para passar para o dataset)
# # Modificar de forma aos caracteres que estão presentes variar para cada fonte.

# prob_chosen = 0.8
# font_masks_ = rd.choice((True, False), size=(n_fonts, N_CHARACTERS), p=[prob_chosen, 1 - prob_chosen]).astype(bool)

# y_ = [None]*n_fonts

# for font in range(n_fonts):
#     y_[font] = np.zeros((N_CHARACTERS, 48, 48), dtype=np.float32)
#     for char_index in range(N_CHARACTERS):
#         y_[font][char_index] = fonts48_df.loc[font][char_index]

# with open('y.dat', 'wb') as f:
#     pickle.dump(y_, f)

# with open('font_masks.dat', 'wb') as f:
#     pickle.dump(font_masks_, f)

# fonts48_df = pd.read_pickle('./data/fonts48_df.pkl');
# fonts48_df.drop(['font_name'], axis=1, inplace=True)

# n_fonts = fonts48_df.shape[0]

# assert fonts48_df.shape[1] == N_CHARACTERS

# Pôr mais bonito, sem loops (Tive que criar lista para ser mais facil criar o dataframe para passar para o dataset)
# Modificar de forma aos caracteres que estão presentes variar para cada fonte.

# prob_chosen = 0.8
# font_masks_ = rd.choice((True, False), size=(n_fonts, N_CHARACTERS), p=[prob_chosen, 1 - prob_chosen]).astype(bool)

# y_ = [None]*n_fonts

# for font in range(n_fonts):
    # y_[font] = np.zeros((N_CHARACTERS, 48, 48), dtype=np.float32)
    # for char_index in range(N_CHARACTERS):
        # y_[font][char_index] = fonts48_df.loc[font][char_index]


# side_pixels = y_[0][0].shape[0]

# n_pixels = side_pixels**2
