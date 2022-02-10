import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x, concat_with):
        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]

        upsampled_x = F.interpolate(
            x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True
        )
        upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)

        upsampled_x = self.convA(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)

        return upsampled_x

class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self, encoder_pretrained=True):
        super(Encoder, self).__init__()       
        self.original_model = models.densenet169(pretrained=encoder_pretrained)

    def forward(self, x):
        features = [x]

        for key, value in self.original_model.features._modules.items(): 
            features.append(value(features[-1]))

        return features

class DenseDepth(nn.Module):
    def __init__(self, encoder_pretrained=True):
        super(DenseDepth, self).__init__()
        self.encoder = Encoder(encoder_pretrained=encoder_pretrained)
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))




class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, t):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(t).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(t)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class CondUpSample(nn.Sequential):
    def __init__(self, skip_input, output_features, num_classes, normalizer):
        super(CondUpSample, self).__init__()
        self.normalizerA = normalizer(skip_input, num_classes, bias=True)
        self.normalizerB = normalizer(output_features, num_classes, bias=True)
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x, concat_with, t):
        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]

        upsampled_x = F.interpolate(
            x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True
        )
        upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)

        upsampled_x = self.normalizerA(upsampled_x, t)
        upsampled_x = self.convA(upsampled_x)
        upsampled_x = self.normalizerB(upsampled_x, t)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.normalizerB(upsampled_x, t)
        upsampled_x = self.leakyrelu(upsampled_x)

        return upsampled_x


class CondUpSamplePlus(nn.Sequential):
    def __init__(self, skip_input, output_features, num_classes, normalizer):
        super(CondUpSamplePlus, self).__init__()
        self.normalizerA = normalizer(skip_input, num_classes, bias=True)
        self.normalizerB = normalizer(output_features, num_classes, bias=True)
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.convC = nn.Conv2d(skip_input, output_features, kernel_size=1, stride=1)


    def forward(self, x, concat_with, t):
        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]

        upsampled_x = F.interpolate(
            x, size=[concat_h_dim, concat_w_dim], mode="nearest"#, align_corners=True
        )
        upx = torch.cat([upsampled_x, concat_with], dim=1)

        upx = self.normalizerA(upx, t)
        upsampled_x = self.convA(upx)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.normalizerB(upsampled_x, t)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        upx = self.convC(upx)
        return upsampled_x + upx

class CondDecoder(nn.Module):
    def __init__(self, num_classes, num_features=1664, decoder_width=1.0):
        super(CondDecoder, self).__init__()
        features = int(num_features * decoder_width)
        res_dim = int(32 * decoder_width)
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.normalizer = ConditionalInstanceNorm2d
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.up1 = CondUpSample(skip_input=features//1 + 256, output_features=features//2, num_classes=num_classes, normalizer=self.normalizer)
        self.up2 = CondUpSample(skip_input=features//2 + 128,  output_features=features//4, num_classes=num_classes, normalizer=self.normalizer)
        self.up3 = CondUpSample(skip_input=features//4 + 64,  output_features=features//8, num_classes=num_classes, normalizer=self.normalizer)
        self.up4 = CondUpSample(skip_input=features//8 + 64,  output_features=features//16, num_classes=num_classes, normalizer=self.normalizer)

        self.conv3 = nn.Conv2d(features//16 + 1, 1, kernel_size=3, stride=1, padding=1)

        self.convy1 = nn.Conv2d(1, res_dim, kernel_size=3, stride=1, padding=1)
        self.normalize1 = self.normalizer(1, num_classes)
        self.normalize2 = self.normalizer(res_dim, num_classes)
        self.convy2 = nn.Conv2d(res_dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features, y, t):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3, t)
        x_d2 = self.up2(x_d1, x_block2, t)
        x_d3 = self.up3(x_d2, x_block1, t)
        x_d4 = self.up4(x_d3, x_block0, t)

        yr = self.normalize1(y, t)
        yr = self.leakyrelu(yr)
        yr = self.convy1(yr)
        yr = self.normalize2(yr, t)
        yr = self.leakyrelu(yr)
        yr = self.convy2(yr)
        y = y + yr

        x_d4 = torch.cat([x_d4, y], dim=1)

        return self.conv3(x_d4)


class CondEncoder(nn.Module):
    def __init__(self, encoder_pretrained=True):
        super(CondEncoder, self).__init__()
        self.original_model = models.densenet169(pretrained=encoder_pretrained)

    def forward(self, x):
        features = [x]

        for key, value in self.original_model.features._modules.items():
            features.append(value(features[-1]))

        return features

class CondDenseDepth(nn.Module):
    def __init__(self, num_classes, encoder_pretrained=True):
        super(CondDenseDepth, self).__init__()
        self.encoder = CondEncoder(encoder_pretrained=encoder_pretrained)
        self.decoder = CondDecoder(num_classes=num_classes)

    def forward(self, x, y, t):
        return self.decoder(self.encoder(x), y, t)



class CondDecoder2(nn.Module):
    def __init__(self, num_classes, num_features=1664, decoder_width=1.0):
        super(CondDecoder2, self).__init__()
        features = int(num_features * decoder_width)
        res_dim = int(32 * decoder_width)
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.normalizer = ConditionalInstanceNorm2d
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.up1 = CondUpSample(skip_input=features//1 + 256, output_features=features//2, num_classes=num_classes, normalizer=self.normalizer)
        self.up2 = CondUpSample(skip_input=features//2 + 128,  output_features=features//4, num_classes=num_classes, normalizer=self.normalizer)
        self.up3 = CondUpSample(skip_input=features//4 + 64,  output_features=features//8, num_classes=num_classes, normalizer=self.normalizer)
        self.up4 = CondUpSample(skip_input=features//8 + 64,  output_features=features//16, num_classes=num_classes, normalizer=self.normalizer)
        self.up5 = UpSample(skip_input=features//16+2,  output_features=features//16+2)

        self.conv3 = nn.Conv2d(features//16 + 2, 2, kernel_size=3, stride=1, padding=1)

        self.convy1 = nn.Conv2d(2, res_dim, kernel_size=3, stride=1, padding=1)
        self.normalize1 = self.normalizer(2, num_classes)
        self.normalize2 = self.normalizer(res_dim, num_classes)
        self.convy2 = nn.Conv2d(res_dim, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, features, y, t):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3, t)
        x_d2 = self.up2(x_d1, x_block2, t)
        x_d3 = self.up3(x_d2, x_block1, t)
        x_d4 = self.up4(x_d3, x_block0, t)

        yr = self.normalize1(y, t)
        yr = self.leakyrelu(yr)
        yr = self.convy1(yr)
        yr = self.normalize2(yr, t)
        yr = self.leakyrelu(yr)
        yr = self.convy2(yr)
        y = y + yr

        x_d5 = self.up5(x_d4, y)

        return self.conv3(x_d5)



class CondDecoderPlus(nn.Module):
    def __init__(self, num_classes, num_features=1664, decoder_width=1.0):
        super(CondDecoderPlus, self).__init__()
        features = int(num_features * decoder_width)
        res_dim = int(32 * decoder_width)
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.normalizer = ConditionalInstanceNorm2d
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.up1 = CondUpSamplePlus(skip_input=features//1 + 256, output_features=features//2, num_classes=num_classes, normalizer=self.normalizer)
        self.up2 = CondUpSamplePlus(skip_input=features//2 + 128,  output_features=features//4, num_classes=num_classes, normalizer=self.normalizer)
        self.up3 = CondUpSamplePlus(skip_input=features//4 + 64,  output_features=features//8, num_classes=num_classes, normalizer=self.normalizer)
        self.up4 = CondUpSamplePlus(skip_input=features//8 + 64,  output_features=features//16, num_classes=num_classes, normalizer=self.normalizer)
        self.up5 = CondUpSamplePlus(skip_input=features//16+2,  output_features=features//16+2, num_classes=num_classes, normalizer=self.normalizer)

        self.conv3 = nn.Conv2d(features//16 + 2, 2, kernel_size=3, stride=1, padding=1)

        self.convy1 = nn.Conv2d(2, res_dim, kernel_size=3, stride=1, padding=1)
        self.normalize1 = self.normalizer(2, num_classes)
        self.normalize2 = self.normalizer(res_dim, num_classes)
        self.convy2 = nn.Conv2d(res_dim, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, features, y, t):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))

        x_d1 = self.up1(x_d0, x_block3, t)
        x_d2 = self.up2(x_d1, x_block2, t)
        x_d3 = self.up3(x_d2, x_block1, t)
        x_d4 = self.up4(x_d3, x_block0, t)

        yr = self.normalize1(y, t)
        yr = self.leakyrelu(yr)
        yr = self.convy1(yr)
        yr = self.normalize2(yr, t)
        yr = self.leakyrelu(yr)
        yr = self.convy2(yr)
        y = y + yr

        x_d5 = self.up5(x_d4, y, t)

        return self.conv3(x_d5)


class CondDenseColor(nn.Module):
    def __init__(self, num_classes, encoder_pretrained=True):
        super(CondDenseColor, self).__init__()
        self.encoder = CondEncoder(encoder_pretrained=encoder_pretrained)
        self.decoder = CondDecoder2(num_classes=num_classes)

    def forward(self, x, y, t):
        return self.decoder(self.encoder(x), y, t)


class CondDenseColorPlus(nn.Module):
    def __init__(self, num_classes, encoder_pretrained=True):
        super(CondDenseColorPlus, self).__init__()
        self.encoder = CondEncoder(encoder_pretrained=encoder_pretrained)
        self.decoder = CondDecoderPlus(num_classes=num_classes)

    def forward(self, x, y, t):
        return self.decoder(self.encoder(x), y, t)