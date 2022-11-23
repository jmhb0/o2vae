import torch.nn as nn

activations = dict(tanh=nn.Tanh, lrelu=nn.LeakyReLU, relu=nn.ReLU, 
    elu=nn.ELU, none=nn.Identity, )

def make_cnn_layers(in_channels=1, hidden_dims=[2,4,8,16,32,64], activation='relu',
                    do_norm_layer=True, kernel_size=3, stride=2,padding=1,
                    **kwargs,
                   ):
    """
    """
    NormLayer = nn.BatchNorm2d if do_norm_layer else nn.Identity
    ActivationLayer = activations[activation]

    modules = []
    for h_dim in hidden_dims:
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            NormLayer(h_dim),
            ActivationLayer(),
        ))
        in_channels = h_dim

    cnn_layer = nn.Sequential(*modules)
    return cnn_layer

class CnnEncoder(nn.Module):
    def __init__(self, n_channels, n_classes, c_dims=[2,4,8,16,32,64], 
                 activation='elu', do_norm_layer=True, kernel_size=3, stride=2, padding=1,
                 **kwargs
                 ):
        """
        Simple CNN encoder with len(c_dims) layers. 
        
        If using the default kernel_size, stride, and padding, then eEach layer halves 
        the size of the featuremap. It's assumed that the feature maps are 1x1 size after 
        the downsampling, otherwise it will error. 
        
        n_channels: imput channels. 
        n_classes: number of output units. If it's a VAE encoder, then it's zdim*2.
        """
        super().__init__()
        
        self.cnn_layers = make_cnn_layers(in_channels=n_channels, hidden_dims=c_dims, 
                                          kernel_size=kernel_size, stride=stride, padding=padding,
                                          activation=activation, do_norm_layer=do_norm_layer)
        self.fc_layer = nn.Linear(1*1*c_dims[-1], n_classes)             # expect 1*1 feature maps
                

    def forward(self, x):
        x = self.cnn_layers(x) # for 64
        n, c, _, _ = x.shape
        x = x.view(n,-1)
        x = self.fc_layer(x)
        
        return x
