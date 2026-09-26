import torch
import torch.nn as nn


NORM_LAYERS = {
    "bn": nn.BatchNorm2d,
    "in": nn.InstanceNorm2d,
    "ln": nn.LayerNorm,
}


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_fea,
        out_fea,
        kernel_size=5,
        stride=1,
        padding=2,
        norm="bn",
        relu_slope=0.2,
        dropout=None,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_fea,
                out_fea,
                kernel_size,
                stride,
                padding,
            )
        ]

        if norm in NORM_LAYERS:
            layers.append(
                NORM_LAYERS[norm](out_fea)
            )

        layers.append(
            nn.LeakyReLU(
                relu_slope,
                inplace=True
            )
        )

        if dropout:
            layers.append(
                nn.Dropout2d(dropout)
            )

        self.layers = nn.Sequential(*layers)


    def forward(self,x):
        return self.layers(x)



class UNetDecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self,x):

        x=self.upsample(x)
        x=self.conv(x)

        return x



class InversionNet(nn.Module):

    """
    OpenBreastUS full-resolution adaptation

    Input:
        [B,2,256,256]

    Output:
        [B,1,256,256]

    Based on InversionNet architecture
    """

    def __init__(
        self,
        dim1=64,
        dim2=128,
        dim3=256,
        dim4=512,
        dim5=1024,
    ):

        super().__init__()


        # Encoder

        self.convblock1 = ConvBlock(
            2,
            dim1,
            kernel_size=5,
            stride=2,
            padding=2
        )


        self.convblock2_1 = ConvBlock(
            dim1,
            dim2,
            kernel_size=5,
            padding=2
        )

        self.convblock2_2 = ConvBlock(
            dim2,
            dim2,
            kernel_size=5,
            stride=2,
            padding=2
        )


        self.convblock3_1 = ConvBlock(
            dim2,
            dim3,
            kernel_size=5,
            padding=2
        )

        self.convblock3_2 = ConvBlock(
            dim3,
            dim3,
            kernel_size=5,
            stride=2,
            padding=2
        )


        self.convblock4_1 = ConvBlock(
            dim3,
            dim4,
            kernel_size=5,
            padding=2
        )

        self.convblock4_2 = ConvBlock(
            dim4,
            dim4,
            kernel_size=5,
            stride=2,
            padding=2
        )


        self.convblock5_1 = ConvBlock(
            dim4,
            dim5,
            stride=2
        )

        self.convblock5_2 = ConvBlock(
            dim5,
            dim5
        )


        self.convblock6 = ConvBlock(
            dim5,
            dim5,
            stride=2
        )


        # Decoder
        #
        # 256 input:
        #
        # 256
        # 128
        # 64
        # 32
        # 16
        # 8
        # 4
        #
        # Decoder:
        #
        # 4 -> 8
        # 8 ->16
        #16 ->32
        #32 ->64
        #64 ->128
        #128->256


        self.decoder6 = UNetDecoderBlock(
            dim5,
            dim4
        )

        self.decoder5 = UNetDecoderBlock(
            dim4,
            dim3
        )

        self.decoder4 = UNetDecoderBlock(
            dim3,
            dim2
        )

        self.decoder3 = UNetDecoderBlock(
            dim2,
            dim1
        )

        self.decoder2 = UNetDecoderBlock(
            dim1,
            64
        )

        self.decoder1 = UNetDecoderBlock(
            64,
            32
        )


        self.final_conv = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )



    def forward(self,x):

        x=self.convblock1(x)

        x=self.convblock2_1(x)
        x=self.convblock2_2(x)

        x=self.convblock3_1(x)
        x=self.convblock3_2(x)

        x=self.convblock4_1(x)
        x=self.convblock4_2(x)

        x=self.convblock5_1(x)
        x=self.convblock5_2(x)

        x=self.convblock6(x)


        x=self.decoder6(x)
        x=self.decoder5(x)
        x=self.decoder4(x)
        x=self.decoder3(x)
        x=self.decoder2(x)
        x=self.decoder1(x)


        x=self.final_conv(x)

        return x