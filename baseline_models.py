# -*- coding = utf-8 -*-
# @File : baseline_models.py
# @Software : PyCharm
# -*- coding = utf-8 -*-
# @File : model.py
# @Software : PyCharm
# 使用PyTorch实现SVM模型
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchvision import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SVM(nn.Module):
    def __init__(self, batch_size, channel_num, img_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(img_size * img_size * channel_num, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNN(nn.Module):
    def __init__(self, channal_num, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(channal_num, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class GRUClassifier(nn.Module):
    def __init__(self, input_shape, hidden_size, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 定义 GRU 层
        self.gru = nn.GRU(input_shape[1] * input_shape[0], hidden_size, num_layers=2, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_shape[1], self.input_shape[0] * self.input_shape[2])
        # 经过 GRU 层，得到最后一个时间步的输出
        output, _ = self.gru(x)
        # 取最后一个时间步的输出作为代表整个序列的特征
        h_n = output[:, -1, :]
        # 经过全连接层，得到分类结果
        out = self.fc(h_n)
        return out




class GAM(nn.Module):
    def __init__(self, channels, rate=4):
        super(GAM, self).__init__()
        mid_channels = channels // rate

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att
        # spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size, channels, patch_size, num_classes, model_dim,
                 emb_dropout=0.0, pool='cls', dim_head=64, channel_rate=2,
                 depth=16, heads=8, mlp_dim=8, dropout=0.0):
        super(ViTtest_origin, self).__init__()

        mid_channels = channels // channel_rate

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_size = patch_size
        self.num_classes = num_classes

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, model_dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.weight = torch.randn(patch_dim, model_dim).to(DEVICE)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, model_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8).to(DEVICE)

        self.transformer = Transformer(model_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
        )

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x = x * x_spatial_att

        # channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        # step 1 convert image to embedding vector sequence
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # step 2 prepend CLS token embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # step3 add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # step4 pass embedding to Transformer Encoder
        # x = self.transformer_encoder(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # step5 do classification
        x = self.to_latent(x)
        out = self.mlp_head(x)

        return out
