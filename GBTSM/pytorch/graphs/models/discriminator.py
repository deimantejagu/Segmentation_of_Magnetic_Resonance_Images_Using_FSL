import torch
import torch.nn as nn

from graphs.models.custom_functions.weight_norm import WN_Conv3d, WN_ConvTranspose3d

class RelationNetwork3D(nn.Module):
    def __init__(self, in_channels=64):
        super(RelationNetwork3D, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(128, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.upsample = nn.ConvTranspose3d(16, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  
            nn.Flatten(),
            nn.Linear(1, 1)  
        )

    def forward(self, x):
        out = self.layer1(x)  
        # Generate relation map
        relation_map = self.layer2(out)  
        out = self.upsample(relation_map)  
        out = self.final_conv(out)  
        # Generate relation score
        logits = self.classifier(out)
        return logits, relation_map

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.input_channel = config.num_modalities
        self.num_classes = config.num_classes
        out_channels = 32

        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout3d(p=0.2)
        self.pool = nn.AvgPool3d(2)

        # Encoder
        self.enc0 = WN_Conv3d(self.input_channel, out_channels, (3,3,3))
        self.enc1 = WN_Conv3d(out_channels, out_channels, (3,3,3))

        self.enc2 = WN_Conv3d(out_channels, out_channels*2, (3,3,3))
        self.enc3 = WN_Conv3d(out_channels*2, out_channels*2, (3,3,3))

        self.enc4 = WN_Conv3d(out_channels*2, out_channels*4, (3,3,3))
        self.enc5 = WN_Conv3d(out_channels*4, out_channels*4, (3,3,3))

        self.enc6 = WN_Conv3d(out_channels*4, out_channels*8, (3,3,3))
        self.enc7 = WN_Conv3d(out_channels*8, out_channels*8, (3,3,3))

        # Decoder
        self.dec1 = WN_ConvTranspose3d(out_channels*8, out_channels*4, (2,2,2), (2,2,2))
        self.dec2 = WN_ConvTranspose3d(out_channels*4, out_channels*2, (2,2,2), (2,2,2))
        self.dec3 = WN_ConvTranspose3d(out_channels*2, out_channels, (2,2,2), (2,2,2))

        self.conv_dec1 = WN_Conv3d(out_channels*8, out_channels*4, (3,3,3))
        self.conv_dec2 = WN_Conv3d(out_channels*4, out_channels*2, (3,3,3))
        self.conv_dec3 = WN_Conv3d(out_channels*2, out_channels, (3,3,3))

        # Final convolution to get the segmentation map
        self.final_conv = WN_Conv3d(out_channels, self.num_classes, (1,1,1))
        # Final activation function to get the probabilities for each class
        self.final_activation = nn.Softmax(dim=1)

        # Relation Network, which takes the concatenated features from the encoder and evaluates real/fake inputs
        self.relation_net = RelationNetwork3D(in_channels=out_channels*2)

    def encode_features(self, x):
        c0 = self.lrelu(self.enc0(x))  
        c1 = self.lrelu(self.enc1(c0)) 
        p1 = self.pool(c1)  

        c2 = self.lrelu(self.enc2(p1))  
        c3 = self.lrelu(self.enc3(c2))  
        p3 = self.pool(c3)  

        c4 = self.lrelu(self.enc4(p3))  
        c5 = self.lrelu(self.enc5(c4))  
        p5 = self.pool(c5) 

        c6 = self.lrelu(self.enc6(p5))  
        c7 = self.lrelu(self.enc7(c6)) 

        # Return the features
        return c7, [c1, c3, c5]

    def decode_features(self, features, skips):
        c7 = features
        c1, c3, c5 = skips

        d1 = self.dec1(c7)
        d1 = torch.cat((d1, c5), dim=1)
        d1 = self.lrelu(self.conv_dec1(d1))

        d2 = self.dec2(d1)
        d2 = torch.cat((d2, c3), dim=1)
        d2 = self.lrelu(self.conv_dec2(d2))

        d3 = self.dec3(d2)
        d3 = torch.cat((d3, c1), dim=1)
        d3 = self.lrelu(self.conv_dec3(d3))

        # Final convolution to get the segmentation map
        seg_map = self.final_conv(d3)
        return seg_map

    def forward(self, x, mode='segment'):
        features, skips = self.encode_features(x)
        if mode == 'segment':
            seg_logits = self.decode_features(features, skips)
            seg_probs = self.final_activation(seg_logits)
            return seg_logits, seg_probs
        elif mode == 'features':
            return skips[0]
        else:
            raise ValueError(f"Unsupported mode: {mode}") 
        
    def forward_relation(self, real_input, fake_input):
        # Get features from the encoder for both real and fake inputs
        real_feats = self.forward(real_input, mode='features')
        fake_feats = self.forward(fake_input, mode='features')

        # Concatenate the features from both inputs
        combined_feats = torch.cat([real_feats, fake_feats], dim=1)
        
        # Pass the combined features through the relation network
        logits, relation_map = self.relation_net(combined_feats)

        # Apply sigmoid to get probabilities for real/fake classification ([0, 1] interval)
        probs = torch.sigmoid(logits)
        return logits, probs, relation_map