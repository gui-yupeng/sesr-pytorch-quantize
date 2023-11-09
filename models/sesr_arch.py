import torch
from torch import nn
from models.model_utils_pt import *
import pdb
from collections import OrderedDict

class inception_sesr(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 full_channels=16,
                 tiny_channels=8,
                 num_lblocks=3,
                 conv_scale=3,
                 single_path=False,
                 scaling_factor=4):
        super().__init__()
        self.tiny_channels  = tiny_channels
        self.conv_scale     = conv_scale
        self.single_path    = single_path
        self.conv_first_p1 = CollapsibleLinearBlock(in_channels=in_channels, out_channels=tiny_channels,
                                                        tmp_channels=512, kernel_size=5, activation='relu')
        self.conv_first_p2 = CollapsibleLinearBlock(in_channels=in_channels, 
                                                    out_channels=tiny_channels+4,
                                                    tmp_channels=256, kernel_size=5, activation='relu')
        self.conv_first_p3 = CollapsibleLinearBlock(in_channels=in_channels, 
                                                    out_channels=tiny_channels+8,
                                                    tmp_channels=256, kernel_size=5, activation='relu')

        residual_layers_p1 = [
            ResidualCollapsibleLinearBlock(in_channels=tiny_channels, out_channels=tiny_channels,
                                                tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        residual_layers_p2 = [
            ResidualCollapsibleLinearBlock(in_channels=tiny_channels+4, out_channels=tiny_channels+4,
                                                tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        residual_layers_p3 = [
            ResidualCollapsibleLinearBlock(in_channels=tiny_channels+8, out_channels=tiny_channels+8,
                                                tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]

        self.residual_block_p1 = nn.Sequential(*residual_layers_p1)
        self.residual_block_p2 = nn.Sequential(*residual_layers_p2)
        self.residual_block_p3 = nn.Sequential(*residual_layers_p3)

        self.add_residual = AddOp()

        self.conv_last_p1 = CollapsibleLinearBlock(in_channels=tiny_channels,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')
        self.last_conv_p2 = CollapsibleLinearBlock(in_channels=tiny_channels+4,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')
        self.last_conv_p3 = CollapsibleLinearBlock(in_channels=tiny_channels+8,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')
        self.add_upsampled_input = AddOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def collapse(self):
        self.conv_first.collapse()
        for layer in self.residual_block:
            layer.collapse()
        self.conv_last.collapse()

    def before_quantization(self):
        self.collapse()

    def forward(self, input):
        initial_features1 = self.conv_first_p1(input)  # Extract features from conv-first
        initial_features2 = self.conv_first_p2(input)  # Extract features from conv-first
        initial_features3 = self.conv_first_p3(input)  # Extract features from conv-first
        residual_features_1 = self.residual_block_p1(initial_features1)  # Get residual features with `lblocks`
        residual_features_1 = self.add_residual(residual_features_1, initial_features1)  # Add init_features and residual
        residual_features_2 = self.residual_block_p2(initial_features2)  # Get residual features with `lblocks`
        residual_features_2 = self.add_residual(residual_features_2, initial_features2)  # Add init_features and residual 
        residual_features_3 = self.residual_block_p3(initial_features3)  # Get residual features with `lblocks`
        residual_features_3 = self.add_residual(residual_features_3, initial_features3)  # Add init_features and residual
        #if self.conv_scale==1:    
        p1= self.conv_last_p1(residual_features_1)  # Get final features from conv-last
        p2= self.last_conv_p2(residual_features_2)  # Get final features from conv-last
        p3= self.last_conv_p3(residual_features_3)  # Get final features from conv-last
        # output = self.add_upsampled_input(final_features, upsampled_input)  # Add final_features and upsampled_input
        if self.single_path:
            if self.conv_scale==1:
                output=p1
            elif self.conv_scale==2:
                output=p2
            elif self.conv_scale==3:
                output=p3
        else:
            output = self.add_upsampled_input(p1, p2)
            output = self.add_upsampled_input(output, p3)
        return self.depth_to_space(output)  # Depth-to-space and return


class split_sesr(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 full_channels=16,
                 tiny_channels=8,
                 num_lblocks=3,
                 conv_scale=3,
                 scaling_factor=4):
        super().__init__()
        self.tiny_channels  = tiny_channels
        self.conv_scale     = conv_scale
        self.conv_first = CollapsibleLinearBlock(in_channels=in_channels, out_channels=tiny_channels,
                                                        tmp_channels=256, kernel_size=5, activation='relu')
        self.conv_first_2 = CollapsibleLinearBlock(in_channels=in_channels, 
                                                    out_channels=(full_channels-tiny_channels)//2,
                                                    tmp_channels=256, kernel_size=5, activation='relu')
        self.conv_first_3 = CollapsibleLinearBlock(in_channels=in_channels, 
                                                    out_channels=(full_channels-tiny_channels)//2,
                                                    tmp_channels=256, kernel_size=5, activation='relu')

        residual_layers = [
            SplitResidualCollapsibleLinearBlock(in_channels=tiny_channels*2, out_channels=tiny_channels*2,
                                                tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        self.residual_block = nn.Sequential(*residual_layers)

        self.add_residual = AddOp()

        self.conv_last = CollapsibleLinearBlock(in_channels=tiny_channels,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')
        self.last_conv2 = CollapsibleLinearBlock(in_channels=tiny_channels//2,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')
        self.last_conv3 = CollapsibleLinearBlock(in_channels=tiny_channels//2,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')
        self.add_upsampled_input = AddOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def collapse(self):
        self.conv_first.collapse()
        for layer in self.residual_block:
            layer.collapse()
        self.conv_last.collapse()

    def before_quantization(self):
        self.collapse()

    def forward(self, input):
        initial_features = self.conv_first(input)  # Extract features from conv-first
        initial_features2 = self.conv_first_2(input)  # Extract features from conv-first
        initial_features3 = self.conv_first_3(input)  # Extract features from conv-first
        with torch.no_grad():
            init_feat = torch.cat([initial_features,initial_features2,initial_features3],1)
        residual_features = self.residual_block(init_feat)  # Get residual features with `lblocks`
        residual_features = self.add_residual(residual_features, init_feat)  # Add init_features and residual
        
        #if self.conv_scale==1:    
        output = self.conv_last(residual_features[:,0                    :self.tiny_channels               ,:,:])  # Get final features from conv-last
        output+= self.last_conv2(residual_features[:,self.tiny_channels             :self.tiny_channels+self.tiny_channels//2   ,:,:])  # Get final features from conv-last
        output+= self.last_conv3(residual_features[:,self.tiny_channels+self.tiny_channels//2 :                       ,:,:])  # Get final features from conv-last
        
        # output = self.add_upsampled_input(final_features, upsampled_input)  # Add final_features and upsampled_input

        return self.depth_to_space(output)  # Depth-to-space and return


class AnchorOp(nn.Module):
    """
    Repeat interleaves the input scaling_factor**2 number of times along the channel axis.
    """
    def __init__(self, scaling_factor, in_channels=3, init_weights=True, freeze_weights=True, kernel_size=1, **kwargs):
        """
        Args:
            scaling_factor: Scaling factor
            init_weights:   Initializes weights to perform nearest upsampling (Default for Anchor)
            freeze_weights:         Whether to freeze weights (if initialised as nearest upsampling weights)
        """
        super().__init__()

        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=(in_channels * scaling_factor**2),
                             kernel_size=kernel_size,
                             **kwargs)
        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, ii % num_channels_per_group,
                kernel_size // 2, kernel_size // 2] = 1.

            new_state_dict = OrderedDict({'weight': weight, 'bias': bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)
                              
class sesr(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_channels=16,
                 num_lblocks=3,
                 scaling_factor=2):
        super().__init__()
        self.anchor = AnchorOp(scaling_factor)
        self.conv_first = CollapsibleLinearBlock(in_channels=in_channels, out_channels=num_channels,
                                                        tmp_channels=256, kernel_size=5, activation='relu')

        residual_layers = [
            ResidualCollapsibleLinearBlock(in_channels=num_channels, out_channels=num_channels,
                                                  tmp_channels=256, kernel_size=3, activation='relu')
            for _ in range(num_lblocks)
        ]
        self.residual_block = nn.Sequential(*residual_layers)

        self.add_residual = AddOp()

        self.conv_last = CollapsibleLinearBlock(in_channels=num_channels,
                                                out_channels=out_channels * scaling_factor ** 2,
                                                tmp_channels=256, kernel_size=5, activation='identity')

        self.add_upsampled_input = AddOp()
        self.depth_to_space = nn.PixelShuffle(scaling_factor)

    def collapse(self):
        self.conv_first.collapse()
        for layer in self.residual_block:
            layer.collapse()
        self.conv_last.collapse()

    def before_quantization(self):
        self.collapse()

    def forward(self, input):
        # input2 = self.anchor(input)
        # initial_features = self.conv_first(input)  # Extract features from conv-first
        # residual_features = self.residual_block(initial_features)  # Get residual features with `lblocks`
        # residual_features = self.add_residual(residual_features, initial_features)  # Add init_features and residual
        # output = self.conv_last(residual_features)  # Get final features from conv-last
        # output = self.add_upsampled_input(output, input2)
        # output = self.depth_to_space(output)
        # #output += torch.nn.functional.interpolate(input, size=output.shape[-2:], mode='bicubic', align_corners=False)
        
        #input2 = self.anchor(input)
        initial_features = self.conv_first(input)  # Extract features from conv-first
        residual_features = self.residual_block(initial_features)  # Get residual features with `lblocks`
        residual_features = self.add_residual(residual_features, initial_features)  # Add init_features and residual
        output = self.conv_last(residual_features)  # Get final features from conv-last
        input2 = input.repeat(1,4,1,1)
        for ii in range(4):
            input2[:,ii,:,:] = input[:,0,:,:]
            input2[:,4+ii,:,:] = input[:,1,:,:]
            input2[:,8+ii,:,:] = input[:,2,:,:]
            #input2[:,0+ii*3,:,:] = input[:,0,:,:]
            #input2[:,1+ii*3,:,:] = input[:,1,:,:]
            #input2[:,2+ii*3,:,:] = input[:,2,:,:]
        output = self.add_upsampled_input(output, input2)
        output = self.depth_to_space(output)
        return output  