import torch
from torch import nn
import torch.nn.

N_CHARACTERS = 26

def encoder_block(n_layers, in_channels, out_channels, kernel_size=3, out=False, groups=1):
    "n_layers is the number of blocks of Conv2d + BackNorm2d + ReLU"
    padding = kernel_size // 2
    block = []
    for layer_index in range(n_layers):
        if layer_index > 0:
            in_channels = out_channels
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups))
        block.append(nn.BatchNorm2d(out_channels))
        if out is False or layer_index != n_layers-1:
            block.append(nn.ReLU())
    return nn.Sequential(*block)

def decoder_block(n_layers, in_channels, out_channels, kernel_size=3, out=False, groups=1):
    "n_layers is the number of blocks of Conv2d + BackNorm2d + ReLU"
    padding = kernel_size // 2
    block = []
    out_channels_ = in_channels
    for layer_index in range(n_layers):
        if layer_index == n_layers-1:
            out_channels_ = out_channels 
        block.append(nn.ConvTranspose2d(in_channels, out_channels_, kernel_size=kernel_size, padding=padding, groups=groups))
        block.append(nn.BatchNorm2d(out_channels_))
        if out is False or layer_index != n_layers-1:
            block.append(nn.ReLU())
    return nn.Sequential(*block)
	


# retirar bias no conv2d??
# https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
#Include visualizations of the feature maps!
# semelhante a unet e a segnet
# este codigo é fixe:https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# o que dizer sobre o receptive field?
# usar o softmax por cima de aplicação linear é equivalente a multinomial logistic regression? por mais bem escrito
# Podemos por ordereddict para ficar com os nomes das layers

# n_channels_per_character = {
#     'encoder_1': 2,
#     'encoder_2': 4,
#     'encoder_3': 4, 
#     'encoder_4': 6,
#     }


# Agora já não é necessário ter multiplo de N_CHARACTERS!
n_channels_per_character = {
    'encoder_1': 2,
    'encoder_2': 2,
    'encoder_3': 4, 
    'encoder_4': 4,
    }

n_channels_encoder_1 = N_CHARACTERS*n_channels_per_character['encoder_1']
n_channels_encoder_2 = N_CHARACTERS*n_channels_per_character['encoder_2']
n_channels_encoder_3 = N_CHARACTERS*n_channels_per_character['encoder_3']
n_channels_encoder_4 = N_CHARACTERS*n_channels_per_character['encoder_4']

n_channels_decoder_1 = n_channels_encoder_1
n_channels_decoder_2 = n_channels_encoder_2
n_channels_decoder_3 = n_channels_encoder_3
n_channels_decoder_4 = n_channels_encoder_4

n_channels_mixer_1 = n_channels_encoder_4*3*3 # 3 is the size in pixels of ?? after the encoding
n_channels_mixer_2 = 1024

n_channels_fc_1 = n_channels_encoder_4*3*3
n_channels_fc_2 = 1024
n_channels_fc_3 = 256
n_channels_fc_4 = 56

# começar a usar parâmetros partilhados nas convoluções a partir de certo momento?

class EnergyNet(nn.Module):
    def __init__(self, inference_mode=False):
        super(EnergyNet, self).__init__()
        self._saved_encoding = None

        #Encoders
        self.encode_conv_1 = encoder_block(2, N_CHARACTERS, n_channels_encoder_1, kernel_size=5, groups=1)
        self.encode_conv_2 = encoder_block(2, n_channels_encoder_1, n_channels_encoder_2, kernel_size=3, groups=1)
        self.encode_conv_3 = encoder_block(2, n_channels_encoder_2, n_channels_encoder_3, kernel_size=3, groups=1)
        self.encode_conv_4 = encoder_block(3, n_channels_encoder_3, n_channels_encoder_4, kernel_size=3, groups=1)

        # Mixers (this and the next decoding ones are just for the init_net)
        self.mix = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_channels_mixer_1, out_features=n_channels_mixer_2),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=n_channels_mixer_2, out_features=n_channels_mixer_1),
            # nn.Dropout(0.3),
            nn.ReLU(),
        )

        
        # Decoders
        self.decode_conv_4 = decoder_block(3, n_channels_decoder_4, n_channels_decoder_3, kernel_size=3, groups=1)
        self.decode_conv_3 = decoder_block(2, n_channels_decoder_3, n_channels_decoder_2, kernel_size=3, groups=1)
        self.decode_conv_2 = decoder_block(2, n_channels_decoder_2, n_channels_decoder_1, kernel_size=3, groups=1)
        self.decode_conv_1 = decoder_block(2, n_channels_decoder_1, N_CHARACTERS, kernel_size=5, groups=1, out=True)

        # this is the fully connected layer in the end of the usual energy_net
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_channels_fc_1, out_features=n_channels_fc_2),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Softplus(beta=25, threshold=20),
            nn.Linear(in_features=n_channels_fc_2, out_features=n_channels_fc_3),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Softplus(beta=25, threshold=20),
            nn.Linear(in_features=n_channels_fc_3, out_features=n_channels_fc_4),
            # nn.Dropout(0.3),
            # nn.ReLU(),
            nn.Softplus(beta=25, threshold=20),
            # It doesn't apply bias since it's not envolved in the differentiation of the energy in order to y
            nn.Linear(in_features=n_channels_fc_4, out_features=1, bias=False),  
            nn.Softplus(beta=25, threshold=20),                 
        )

    def forward(self, x, mode=None, save_encoding=False):
        if mode is None:
            mode = self.mode
        else:
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
        x, indices_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        # if save_encoding:
        #     # Usar o clone, e o detach?
        #     self.saved_encoding = x.clone()
        # else:
        #     self.saved_encoding = None

        if self.mode == 'init':
            shape_after_encoding = x.shape
            x = self.mix(x)
            
            # Decode (include the dimensions as parameter to give good result? https://github.com/say4n/pytorch-segnet/blob/f7738c6bce384b54fcbb3fe8aff02736d6ec2285/src/model.py#L333)
            x = x.view(shape_after_encoding)
            x = F.max_unpool2d(x, indices_4, kernel_size=2, stride=2)
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
        else:
            raise Exception('Introduce a valid mode!')
    
        return self.output

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
