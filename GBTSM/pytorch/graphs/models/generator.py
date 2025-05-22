import torch
import torch.nn as nn

from graphs.weights_initializer import weights_init
from graphs.models.custom_functions.weight_norm import WN_ConvTranspose3d

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU(inplace=True)
        
        self.sh1, self.sh2, self.sh3, self.sh4 = int(self.config.patch_shape[0]/16), int(self.config.patch_shape[0]/8),\
                                                 int(self.config.patch_shape[0]/4), int(self.config.patch_shape[0]/2)
        
        kernel_size = [2, 2, 2]
        stride = [2, 2, 2]

        # Transform 16 channels relation map to 32 channels
        self.relation_map_conv = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=16, stride=16)

        # Transform noise to 512 channels feature map
        self.linear = nn.Linear(self.config.noise_dim, self.sh1 * self.sh1 * self.sh1 * 512)
        self.batch_norm0 = nn.BatchNorm3d(512 + 32)

        self.deconv1 = nn.ConvTranspose3d(in_channels=512 + 32, out_channels=256, kernel_size=kernel_size, stride=stride)
        self.batch_norm1 = nn.BatchNorm3d(256)

        self.deconv2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride)
        self.batch_norm2 = nn.BatchNorm3d(128)

        self.deconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride)
        self.batch_norm3 = nn.BatchNorm3d(64)

        self.deconv4 = WN_ConvTranspose3d(64, self.config.num_modalities, kernel_size, stride)
        
        self.activation = nn.Tanh()

        self.apply(weights_init)

    def forward(self, z, relation_map):
        out = self.linear(z)
        out = out.view(-1, 512, self.sh1, self.sh1, self.sh1)
        
        relation_map = self.relation_map_conv(relation_map)
        # Concatenate the relation map with the feature map
        out = torch.cat([out, relation_map], dim=1)

        out = self.relu(self.batch_norm0(out))

        out = self.deconv1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        
        out = self.deconv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.deconv3(out)
        out = self.batch_norm3(out)
        out = self.relu(out)

        out = self.deconv4(out)
        out = self.activation(out)

        # Returns generated 3D image
        return out
        