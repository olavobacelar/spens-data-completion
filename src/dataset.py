# Preparar os dados no __init__ para ser mais fácil depois.
#

import pandas as pd
import numpy as np
import numpy.random as rd
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

N_LETTERS = 26  # tirar daqui, precisa de estar também no modulo que chama. E não devia estar hardcoded

@dataclass(frozen=True)
class DatasetConfig:
    '''To configure a Dataset, set those parameters below.

    It's not exactly necessary, but is helpful to have the configuration
    as a dataclass (Google Colab still doesn't type check anything though.)
    '''

    batch_size: int
    dataset_mode: str
    n_visible_letters: Union[None, int] = None
    prob_choice: Union[None, float] = None
    initial_seed: int = 48

    def __post_init__(self):
        # With Google Colab we can't easily use type hints, so we add this
        if self.dataset_mode not in ['fixed', 'mixed', 'prob'] or not isinstance(self.batch_size, int):
            raise ValueError

    def __iter__(self):
        return iter(astuple(self))

    def print_training_repr(self):
        output = f'Dataset Config: batch_size: {self.batch_size} • mode: '
        if self.dataset_mode == 'fixed':
            print(output + f'fixed • n_visible_letters: {self.n_visible_letters}')
        elif self.dataset_mode == 'mixed':
            print(output + f'mixed')
        elif self.dataset_mode == 'prob':
            print(output + f'prob • prob_choice: {self.prob_choice}')

class FontsDataset(Dataset):
    # É ineficiente transformar em listas??? Tentar mudar para tensores mesmo?
    def __init__(self, data, dataset_config):
        # y is the (unmasked) output. We'll need to construct the masked input
        # # Não uso o mode, que passava antes. Apagar?

        _, dataset_mode, n_visible_letters, prob_choice, initial_seed = dataset_config

        if dataset_mode not in ['fixed', 'mixed', 'prob']:
            raise ValueError("the variable mode must be one of the following: "
                             "'fixed', 'mixed', 'prob'")

        # Data input and self.letter_masks are expected to be a pandas dataframe
        self.y = data.y.values.tolist()
        # Cuidado com as masks, pode ser lista de Nones?
        if self.y[0].shape[0] != N_LETTERS:
            raise ValueError
        self.n_fonts = len(self.y)

        # Here we build a random number generator for each dataset, which adds to
        # the ease of reproducilibity

        self.init_rngs(initial_seed)

        self.build_masked_input(dataset_config)
    
    def __len__(self):
        return self.n_fonts
  
    def __getitem__(self, index):
        # É mais rápido transformar em tensor e float aqui ou dentro da rotina das features?
        return torch.tensor(self.x[index]), torch.tensor(self.y[index]), self.letter_masks[index]

    def init_rngs(self, seed):
        self.rng = random.Random(seed)
        self.np_rng = rd.default_rng(seed)
    
    def get_rngs_states(self):
        return self.rng.getstate(), self.np_rng.bit_generator.state
    
    def set_rngs_states(self, states):
        self.rng.setstate(states[0])
        self.np_rng.bit_generator.state = states[1]

    def set_seed(self, seed):
        # What exactly does this do?
        # Different modules are used when the probability is used vs. when a fixed n_visible_letters is used
        # therefore, when we call set_seed we set both seeds.
        # Torch.manual_seed will set the seed for the data loader, fixing the order the different fonts are
        # loaded in shuffle mode.
        self.rng.seed(seed)
        self.np_rng = rd.default_rng(seed)

    def new_letter_masks_with_prob(self, n_fonts, prob_choice):
        # prob_choice is the probability of a certain character NOT being masked
        
        assert isinstance(n_fonts, int) and isinstance(prob_choice, float) 
        
        letter_masks = self.random_choice_at_least_one(n_fonts, prob_choice)
        letter_masks = list(letter_masks.astype(bool))
        return letter_masks

    def new_letter_masks_fixed(self, n_fonts, n_visible_letters):
        
        assert isinstance(n_fonts, int) and isinstance(n_visible_letters, int)
        assert 1 <= n_visible_letters <= 25

        letter_masks = np.concatenate([np.ones((n_fonts, n_visible_letters), dtype=bool),
                                       np.zeros((n_fonts, N_LETTERS-n_visible_letters), dtype=bool)],
                                      axis=1)
        
        for one_font_of_letter_masks in letter_masks:
            self.np_rng.shuffle(one_font_of_letter_masks)

        return letter_masks

    def new_letter_masks_mixed(self, n_fonts):
        
        letter_masks = []

        for i in range(n_fonts):
            random_n_letters = self.rng.randint(1, N_LETTERS-1)
            letter_mask = np.concatenate([np.ones(random_n_letters, dtype=bool),np.zeros(N_LETTERS - random_n_letters, dtype=bool)])
            self.np_rng.shuffle(letter_mask)
            letter_masks.append(letter_mask)
        
        return letter_masks

    def build_masked_input(self, dataset_config, another_seed=None, letter_masks=None):
        '''Generate a masked input of the letters for each font, according to
        the dataset_mode chosen. Each letter is masked by a list 'letter_masks', 
        which is generated by other functions. This masks are used to modify
        the input, so that only the letters corresponding to a True value in the
        mask are shown.
        We can also pass another_seed that may be different from the one passed
        in the beginning.
        '''

        # Describe possible modes in the doc string!
		
        _, dataset_mode, n_visible_letters, prob_choice, _ = dataset_config
        
        if dataset_mode not in ['fixed', 'mixed', 'prob']:
            raise ValueError("the variable dataset_mode must be one of the following: "
                             "'fixed', 'mixed', 'prob'")
        
        if another_seed is not None:
            self.set_seed(another_seed)

        # Generate letter_masks depending on the dataset_mode, or get them from a parameter
        if letter_masks is None:
            if dataset_mode == 'fixed':
                if n_visible_letters is not None:
                    self.letter_masks = self.new_letter_masks_fixed(self.n_fonts, n_visible_letters)
                    self.n_visible_letters = n_visible_letters
                else:
                    raise ValueError
            
            elif dataset_mode == 'mixed':
                self.letter_masks = self.new_letter_masks_mixed(self.n_fonts)

            elif dataset_mode == 'prob':
                if prob_choice is not None:
                    self.letter_masks = self.new_letter_masks_with_prob(self.n_fonts, prob_choice)
                    self.prob_choice = prob_choice
                else:
                    raise ValueError
        else:
            self.letter_masks = letter_masks
        
        if len(self.letter_masks) != self.n_fonts:
            raise ValueError('The number of fonts and masks for them is not the same!')

        # Generate a masked input, using the given letter_masks 
        self.x = [None]*self.n_fonts
        for font_index in range(self.n_fonts):
            self.x[font_index] = np.zeros((N_LETTERS, *LETTER_SHAPE), dtype=np.float32) + 0.5
            for letter_index in range(N_LETTERS):
                if self.letter_masks[font_index][letter_index] == True:
                    self.x[font_index][letter_index] = self.y[font_index][letter_index]


    def random_choice_at_least_one(self, n_fonts, prob_choice):
        '''Choose one letter per font randomly, and choose the others ones with
        probability equal to prob_choice. Therefore, choosing prob_choice=0.0
        will select only one letter per font. We do this because we want at
        least one letter per font to be shown during training or testing.
        '''

        def choose_mask(i, random_letter, prob_choice):
            if self.rng.random() < prob_choice or i == random_letter:
                return True
            else:
                return False
       
        letter_masks = []
        for font in range(n_fonts):
            random_letter = self.rng.randint(0, N_LETTERS-1)
            letter_mask = np.array([choose_mask(i, random_letter, prob_choice) for i in range(N_LETTERS)])
            letter_masks.append(letter_mask)

        return np.array(letter_masks)

    def get_letter_masks(self):
        if self.letter_masks is not None:
            return self.letter_masks
        else:
            raise Exception("No font masks were created yet!")
