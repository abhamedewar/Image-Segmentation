from torch import nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down = nn.ModuleList()

        for f in features:
            self.down.append(DoubleConv(in_channels, f))
            in_channels = f

        self.bottleneck = nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1)
        
        self.up = nn.ModuleList()

        for f in features[::-1]:
            self.up.append(nn.ConvTranspose2d(in_channels=f*2, out_channels=f, kernel_size=2, stride=2))
            self.up.append(DoubleConv(in_channels=f*2, out_channels=f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        
        skip_connections = []

        for layer in self.down:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
    
        x = self.bottleneck(x)
        i = len(skip_connections) - 1
        for layer in self.up:
            if isinstance(layer, DoubleConv):
                x = torch.cat((skip_connections[i], x), dim=1)
                i -= 1
                x = layer(x)
            else:
                x = layer(x)
        
        x = self.final_conv(x)
        return x

# #test U-Net
# x = torch.randn(2, 1, 400, 400)
# model = UNet(in_channels=1)
# assert model(x).shape == x.shape
# print(model(x).shape)
# print(x.shape)