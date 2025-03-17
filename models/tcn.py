import torch
import torch.nn as nn
import torch.nn.functional as F

class FrontendCNN(nn.Module):
    def __init__(self, channels=16, dropout=0.1):
        super(FrontendCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ELU(),
            nn.Dropout(dropout),
            # Pool only in frequency dimension
            nn.MaxPool2d(kernel_size=(3, 1)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ELU(),
            nn.Dropout(dropout),
            # Pool only in frequency dimension
            nn.MaxPool2d(kernel_size=(3, 1)),
        )
        
        # So we need a final conv to reduce freq dimension from 9 to 1
        self.final_reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # This will make freq dimension exactly 1
            nn.ELU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.final_reduction(x)  
        
        x = x.squeeze(2)
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TCNBlock, self).__init__()
        
        # Calculate proper padding for same length output
        padding = (kernel_size - 1) * dilation // 2
        
        # First dilated convolution
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        ))
        
        # Second dilated convolution
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels,  # Using out_channels as input
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        ))
        
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Add residual connection if input and output channels differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.elu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        
        # Add residual connection
        x = x + residual
        return x

class NonCausalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_levels, dropout=0.1):
        super(NonCausalTCN, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer: input_channels = out_channels and dilation=1
        self.layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation=1, dropout=dropout))
        
        # Remaining TCN blocks with exponentially increasing dilation
        for i in range(1, num_levels):
            dilation_size = 2 ** i
            self.layers.append(TCNBlock(out_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BeatNet(nn.Module):
    def __init__(self, channels=16, tcn_layers=11, kernel_size=5, dropout=0.1):
        super(BeatNet, self).__init__()
        
        self.frontend = FrontendCNN(channels, dropout)
        self.tcn = NonCausalTCN(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            num_levels=tcn_layers,
            dropout=dropout
        )
        self.output_layer = nn.Conv1d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid() # To get output for each frame in beat probablity
        
    def forward(self, x):
        #print("Input shape:", x.shape)
        
        if x.shape[2] == 81:
            x = x.permute(0, 2, 1)  # [batch_size, freq, time]
            #print("After permute:", x.shape)
        
        x = x.unsqueeze(1)      
        #print("After unsqueeze:", x.shape)
        
        # Frontend CNN processing - reduces frequency dimension to exactly 1
        x = self.frontend(x)    
        #print("After frontend:", x.shape)
        
        # TCN processing along time dimension
        x = self.tcn(x)  
        #print("After TCN:", x.shape)
        
        # Output layer
        x = self.output_layer(x)  # [batch_size, 1, time]
        #print("Before squeeze:", x.shape)
        x = x.squeeze(1)          # [batch_size, time]
        #print("Final output shape:", x.shape)
        
        # Apply sigmoid activation for beat probability
        x = self.sigmoid(x)
        
        return x