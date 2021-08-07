from typing import List, Tuple
from typing import Tuple
from dataclasses import dataclass, astuple
import torch
from torch import nn
import torch.nn.functional as F

N_LETTERS = 26

def encoder_block(n_layers, in_channels, out_channels, kernel_size=3, use_batchnorm=False, use_softplus=True, beta=1.):
    "n_layers is the number of blocks of Conv2d + BackNorm2d + ReLU"
    padding = kernel_size // 2
    block = []
    for layer_index in range(n_layers):
        if layer_index > 0:
            in_channels = out_channels
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_batchnorm:
            block.append(nn.BatchNorm2d(out_channels))
        if use_softplus:
            block.append(nn.Softplus(beta))
        else:
            block.append(nn.ReLU())
    return nn.Sequential(*block)

def decoder_block(n_layers, in_channels, out_channels, kernel_size=3, out=False, use_batchnorm=False, use_softplus=True, beta=1.):
    "n_layers is the number of blocks of Conv2d + BackNorm2d + ReLU"
    padding = kernel_size // 2
    block = []
    out_channels_ = in_channels
    for layer_index in range(n_layers):
        if layer_index == n_layers-1:
            out_channels_ = out_channels 
        block.append(nn.ConvTranspose2d(in_channels, out_channels_, kernel_size=kernel_size, padding=padding))
        if use_batchnorm:
            block.append(nn.BatchNorm2d(out_channels_))
        if out is False or layer_index != n_layers-1:
            if use_softplus:
                block.append(nn.Softplus(beta))
            else:
                block.append(nn.ReLU())
    return nn.Sequential(*block)

@dataclass(frozen=True)
class EnergyNetConfig:
    '''To configure an Energy Network architecture, set those parameters below.

    It's not exactly necessary, but is helpful to have the configuration
    as a dataclass (Google Colab still doesn't type check anything though.)
    '''

    n_channels_conv: Tuple[int]
    n_layers_per_conv_block: Tuple[int]
    kernel_sizes: Tuple[int]
    n_linear_features: Tuple[int]
    dropout_probs: Tuple[float]
    use_batchnorm: bool = False
    beta: float = 1.
    last_softplus: bool = False
    use_softplus: bool = True

    def __post_init__(self):
        # With Google Colab we can't easily use type hints, so we add this
        if not (isinstance(self.n_channels_conv, tuple) and isinstance(self.n_layers_per_conv_block, tuple) and
            isinstance(self.kernel_sizes, tuple) and isinstance(self.n_linear_features, tuple) and
            isinstance(self.dropout_probs, tuple) and isinstance(self.use_batchnorm, bool) and
            isinstance(self.beta, float) and isinstance(self.last_softplus, bool) and
            isinstance(self.use_softplus, bool)):
            raise TypeError('The parameters must be of the given types!')
        if len(self.n_channels_conv) != len(self.n_layers_per_conv_block) or \
            len(self.n_layers_per_conv_block) != len(self.kernel_sizes):
            raise ValueError('n_channels_conv, n_layers_per_conv_block, and '
                              'kernel_sizes must have the same length!')
        if len(self.n_linear_features) != len(self.dropout_probs):
            raise ValueError('n_linear_features and dropout_probs must have the same length!')

    def __iter__(self):
        # Dataclass doesn't unpack by default
        # I could have written that:
        
        # return iter([self.n_channels_conv, self.kernel_sizes, self.n_features_fc,
        #             self.beta, self.use_softplus, self.last_softplus])
        return iter(astuple(self))

    def print_training_repr(self):
        print(f'EnergyNet Config: n_channels_conv: {self.n_channels_conv} • n_layers_per_conv_block: {self.n_layers_per_conv_block} • '
              f'kernel_sizes: {self.kernel_sizes}\n' + ' '*18 + f'n_linear_features: {self.n_linear_features} • '
              f'dropout_probs: {self.dropout_probs} • beta: {self.beta}')

class EnergyNet(nn.Module):
    """Energy Net accepting a configuration object for defining the architecture
    
    TO DO: Maybe generalize for the other energy nets to be used."""

    # transformar todas as relus em softplus!
    def __init__(self, config, architecture='usual', beta_is_parameter=False):
        super(EnergyNet, self).__init__()
        
        if not isinstance(config, EnergyNetConfig):
            raise TypeError('One must insert a valid EnergyNetConfig!')

        self.architecture = architecture
        n_channels_conv, n_layers_per_conv_block, kernel_sizes, n_linear_features, \
        dropout_probs, use_batchnorm, beta, last_softplus, use_softplus = config
        n_conv_blocks = int(len(n_layers_per_conv_block))

        # self.beta = torch.tensor(beta, device=device)
        # if beta_is_parameter:
        #     self.beta = torch.nn.Parameter(self.beta)
        
        softplus = torch.nn.Softplus(beta)

        n_channels_conv = [N_LETTERS] + list(n_channels_conv)

        conv_modules = OrderedDict()
        # self.conv = nn.ModuleList()
        for i, (n_layers, kernel_size) in enumerate(zip(n_layers_per_conv_block, kernel_sizes)):
            # self.conv.append(encoder_block(n_layers, n_channels_conv[i], n_channels_conv[i+1], kernel_size, use_softplus, beta))
            conv_modules['encode_conv_' + str(i+1)] = encoder_block(n_layers, n_channels_conv[i], n_channels_conv[i+1],
                                                                    kernel_size, use_batchnorm, use_softplus, beta)
        self.conv = nn.Sequential(conv_modules)

        # This will compute the number of pixels in the last feature maps after all the cnn transformations
        # n_conv_blocks is the number of blocks, each of which reduces the size of the 
        # feature maps by a half
        side_size_last_feature_map = int(SIDE_SIZE_PIXELS / (2**n_conv_blocks))
        n_pixels_last_feature_map = int(side_size_last_feature_map**2)

        n_linear_features = [n_channels_conv[-1]*n_pixels_last_feature_map] + \
                            list(n_linear_features)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Flatten())
        for layer_idx, dropout_prob in enumerate(dropout_probs):
            self.fc.append(nn.Linear(in_features=n_linear_features[layer_idx], out_features=n_linear_features[layer_idx+1]))
            self.fc.append(nn.Dropout(dropout_prob))
            self.fc.append(softplus)
        # It doesn't apply bias since it's not envolved in the differentiation of the energy in order to y
        self.fc.append(nn.Linear(in_features=n_linear_features[-1], out_features=1, bias=False))

        if last_softplus:
            self.fc.append(softplus)

    def forward(self, x, mode='init'):  # apagar mode

        if self.architecture == 'usual':
            return_indices = False

        for block_idx in range(len(self.conv)):
            x = self.conv[block_idx](x)
            x = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=return_indices)
    
        for op_idx in range(len(self.fc)):
            x = self.fc[op_idx](x)

        return x

class AutoencoderBaseline(nn.Module):
    def __init__(self, architecture='baseline'):
        super(AutoencoderBaseline, self).__init__()
        self._saved_encoding = None

        if architecture not in ['baseline', 'baseline_to_init_unrolled']:
            raise ValueError("The 'architecture' variable must be either "
                             "'baseline' or 'baseline_to_init_unrolled'")

        #Encoders
        self.encode_conv_1 = encoder_block(2, N_LETTERS, n_channels_encoder_1, kernel_size=5,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        self.encode_conv_2 = encoder_block(2, n_channels_encoder_1, n_channels_encoder_2, kernel_size=5,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        self.encode_conv_3 = encoder_block(3, n_channels_encoder_2, n_channels_encoder_3, kernel_size=3,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        self.encode_conv_4 = encoder_block(3, n_channels_encoder_3, n_channels_encoder_4, kernel_size=3,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
     
        # Decoders
        self.decode_conv_4 = decoder_block(3, n_channels_decoder_4, n_channels_decoder_3, kernel_size=3,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        self.decode_conv_3 = decoder_block(3, n_channels_decoder_3, n_channels_decoder_2, kernel_size=3,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        self.decode_conv_2 = decoder_block(2, n_channels_decoder_2, n_channels_decoder_1, kernel_size=5,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        self.decode_conv_1 = decoder_block(2, n_channels_decoder_1, N_LETTERS, kernel_size=5, out=True,
                                           use_batchnorm=use_batchnorm, use_softplus=use_softplus)
        

        # this is the fully connected layer in the end of the usual energy_net.
        # If the right mode is chosen, it will result in having
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_channels_fc_1, out_features=n_channels_fc_2),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Softplus(beta=beta, threshold=20),
            nn.Linear(in_features=n_channels_fc_2, out_features=n_channels_fc_3),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Softplus(beta=beta, threshold=20),
            nn.Linear(in_features=n_channels_fc_3, out_features=1, bias=False),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            # nn.Softplus(beta=25, threshold=20),
            # # It doesn't apply bias since it's not envolved in the differentiation of the energy in order to y
            # nn.Linear(in_features=n_channels_fc_4, out_features=1, bias=False),
            # nn.Softplus(beta=25, threshold=20),                 
        )

        self.softplus = nn.Softplus(beta=beta, threshold=20)

    def forward(self, x, mode=None, save_encoding=False):
        # if mode is None:
        #     mode = self.mode
        if mode is not None:
            self.mode = mode

        # if self.mode == 'init_energy':
        #     return self.fc(self.saved_encoding)

        # Encode
        x = self.encode_conv_1(x)
        x, indices_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = self.encode_conv_2(x)
        x, indices_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = self.encode_conv_3(x)
        x, indices_3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = self.encode_conv_4(x)
        # x, indices_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        # if save_encoding:
        #     # Usar o clone, e o detach?
        #     self.saved_encoding = x.clone()
        # else:
        #     self.saved_encoding = None

        if self.mode == 'init':
            # shape_after_encoding = x.shape
            # x = self.mix(x)

            # Decode (include the dimensions as parameter to give good result? https://github.com/say4n/pytorch-segnet/blob/f7738c6bce384b54fcbb3fe8aff02736d6ec2285/src/model.py#L333)
            # x = x.view(shape_after_encoding)
            # x = F.max_unpool2d(x, indices_4, kernel_size=2, stride=2)
            x = self.decode_conv_4(x)
            x = F.max_unpool2d(x, indices_3, kernel_size=2, stride=2)
            x = self.decode_conv_3(x)
            x = F.max_unpool2d(x, indices_2, kernel_size=2, stride=2)
            x = self.decode_conv_2(x)
            x = F.max_unpool2d(x, indices_1, kernel_size=2, stride=2)
            x = self.decode_conv_1(x)
            self.output = x
            # self.output = torch.sigmoid(x)
        elif self.mode == 'usual':
            # vou precisar de self.encoded_input para mais alguma coisa?
            self.output = self.fc(x)
            if last_softplus:
                self.output = self.softplus(self.output)
        else:
            raise Exception('Introduce a valid mode!')
        return self.output

# Teste da EnergyNet

# energy_net_config_A = EnergyNetConfig(n_channels_conv = (120, 140, 180, 220),
#                           n_layers_per_conv_block = (2, 2, 3, 3),
#                           kernel_sizes = (5, 5, 3, 3),
#                           n_linear_features = (200, 80),
#                           dropout_probs = (0.3, 0.5))

# energy_net = EnergyNet(energy_net_config_A).to(device)

    # @property
    # def saved_encoding(self):
    #     if self._saved_encoding is not None:
    #         return self._saved_encoding
    #     else:
    #         raise Exception('The encoding was not saved last time the energy'
    #                         'function ran!')
    # @saved_encoding.setter
    # def saved_encoding(self, value):
    #     self._saved_encoding = value

        # self.register_y_parameter()
    
    # def register_y_parameter(self):
    #     exist_y_param = hasattr(self, 'y_param')
    #     if not exist_y_param:
    #         # For some reason just including y_param as a parameter in the 
    #         # __init__ was not enough. Had to add it this way:
    #         y_param = nn.Parameter(torch.ones((batch_size, N_CHARACTERS_present, side_pixels, side_pixels)), requires_grad=True)
    #         y_param.to(device)
    #         self.register_parameter('y', y_param)
