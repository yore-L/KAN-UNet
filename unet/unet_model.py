""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from kans import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_kan=False, kan_type='fast', dropout=0.0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        if use_kan:
            if kan_type=='fast':
                self.mlp1 = mlp_fastkan([128, 256, 128], dropout=dropout)
            else:
                self.mlp1 = mlp_kan([128, 256, 128], dropout=dropout)
        else:
            self.mlp1 = Mlp(in_features=128, hidden_features=256, out_features=128)

        self.down2 = Down(128, 256)
        if use_kan:
            if kan_type=='fast':
                self.mlp2 = mlp_fastkan([256, 512, 256], dropout=dropout)
            else:
                self.mlp2 = mlp_kan([256, 512, 256], dropout=dropout)
        else:
            self.mlp2 = Mlp(in_features=256, hidden_features=512, out_features=256)

        self.down3 = Down(256, 512)
        if use_kan:
            if kan_type=='fast':
                self.mlp3 = mlp_fastkan([512, 1024, 512], dropout=dropout)
            else:
                self.mlp3 = mlp_kan([512, 1024, 512], dropout=dropout)
        else:
            self.mlp3 = Mlp(in_features=512, hidden_features=1024, out_features=512)

        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # MLP 块 
        if use_kan:
            if kan_type=='fast':
                self.mlp_bottom = mlp_fastkan([1024//factor, 2048, 1024//factor], dropout=dropout)
            else:
                self.mlp_bottom = mlp_kan([1024//factor, 2048, 1024//factor], dropout=dropout)
        else:
            self.mlp_bottom = Mlp(in_features=1024//factor, hidden_features=2048, out_features=1024//factor)


        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]

        # flatten 空间维度以输入 MLP/KAN
        B, C, H, W = x2.shape
        x2_flat = x2.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        x2_flat = self.mlp1(x2_flat)                # [B, H*W, 128]
        x2 = x2_flat.transpose(1, 2).reshape(B, -1, H, W)  # [B, 128, H/2, W/2]

        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        # 注意 mlp2 接收 [B, H'*W', 256]
        B, C3, H3, W3 = x3.shape
        x3_flat = x3.view(B, C3, -1).transpose(1, 2)
        x3_flat = self.mlp2(x3_flat)
        x3 = x3_flat.transpose(1, 2).reshape(B, -1, H3, W3)

        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        B, C4, H4, W4 = x4.shape
        x4_flat = x4.view(B, C4, -1).transpose(1, 2)
        x4_flat = self.mlp3(x4_flat)
        x4 = x4_flat.transpose(1, 2).reshape(B, -1, H4, W4)

        x5 = self.down4(x4)   # [B, 1024/factor, H/16, W/16]
        B, C5, H5, W5 = x5.shape
        x5_flat = x5.view(B, C5, -1).transpose(1, 2)
        x5_flat = self.mlp_bottom(x5_flat)
        x5 = x5_flat.transpose(1, 2).reshape(B, -1, H5, W5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        # 对 mlp 应用检查点
        self.mlp = torch.utils.checkpoint.checkpoint(self.mlp)
        # self.mlla_block = torch.utils.checkpoint(self.mlla_block)

        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=1, use_kan=True)
    print(model)

    # 测试输入
    x = torch.randn(1, 3, 1024, 1024)  # 假设输入是3通道的1024x1024图像
    output = model(x)
    print(f"Output shape: {output.shape}")