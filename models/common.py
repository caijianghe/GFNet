import torch.nn as nn
import torch
from torchvision import transforms

from models.modules import InvertibleConv1x1
import torch.nn.init as init
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2),
        16: (20, 16, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
 

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class DownBlock(nn.Module):
    def __init__(self, scale, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        down_m = []
        for _ in range(scale):
            down_m.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.PReLU()
                )
            )
        self.downModule = nn.Sequential(*down_m)

    def forward(self, x):
        x = self.downModule(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, scale, in_channels=None, out_channels=None):
        super(UpBlock, self).__init__()
        up_m = []
        for _ in range(scale):
            up_m.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True),
                    nn.PReLU()
                )
            )
        self.downModule = nn.Sequential(*up_m)

    def forward(self, x):
        x = self.downModule(x)
        return x

class FreDiff(nn.Module):
    def __init__(self, channels,rgb_channels):
        super(FreDiff, self).__init__()

        self.fuse_c = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.fuse_sub = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(2*channels,channels,1,1,0)
        self.pre_rgb = nn.Conv2d(rgb_channels,channels,1,1,0)
        self.pre_dep = nn.Conv2d(channels,channels,1,1,0)

        self.sig = nn.Sigmoid()

    def forward(self, dp, rgb):

        dp1 = self.pre_dep(dp)   # 用1*1卷积来放大
        rgb1 = self.pre_rgb(rgb)

        fuse_c = self.fuse_c(dp1)  # dp1做1*1卷积

        fuse_sub = self.fuse_sub(torch.abs(rgb1 - dp1))  # 差值后再做1*1卷积
        cat_fuse = torch.cat([fuse_c,fuse_sub],1)   # 插值后加上原本的，只不过都做了1*1卷积

        return self.post(cat_fuse)

class SDB(nn.Module):
    def __init__(self, channels,rgb_channels):
        super(SDB, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(rgb_channels,rgb_channels,1,1,0)
        self.amp_fuse = FreDiff(channels,rgb_channels)
        self.pha_fuse = FreDiff(channels,rgb_channels)
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, dp, rgb):

        _, _, H, W = dp.shape
        dp = torch.fft.rfft2(self.pre1(dp)+1e-8, norm='backward')
        rgb = torch.fft.rfft2(self.pre2(rgb)+1e-8, norm='backward')
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)
        rgb_amp = torch.abs(rgb)
        rgb_pha = torch.angle(rgb)
        amp_fuse = self.amp_fuse(dp_amp,rgb_amp)  # 幅度做FreDiff
        pha_fuse = self.pha_fuse(dp_pha,rgb_pha)  # 相位做Diff

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))  # 做逆傅里叶变化回来

        return self.post(out)

class get_Fre(nn.Module):
    def __init__(self):
        super(get_Fre, self).__init__()

    def forward(self, dp):

        dp = torch.fft.rfft2(dp, norm='backward')
        dp_amp = torch.abs(dp)     # 获取幅度和相位
        dp_pha = torch.angle(dp)

        return dp_amp, dp_pha

class SFIB(nn.Module):
    def __init__(self, channels, rgb_channels,scale):
        super(SFIB, self).__init__()
        self.rgbprocess = nn.Conv2d(rgb_channels, rgb_channels, 3, 1, 1)
        self.rgbpre = nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, channels + rgb_channels, channels),
                                         nn.Conv2d(channels + rgb_channels, channels, 1, 1, 0))
        self.fre_process = SDB(channels, rgb_channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels, channels, 3, 1, 1)

        self.fuse_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels, channels, 1, 1, 0))

        self.downBlock = DenseProjection(channels, channels, scale, up=False, bottleneck=False)
        self.upBlock = DenseProjection(channels, channels, scale, up=True, bottleneck=False)

    def forward(self, dp, rgb):  # , i

        dp = self.upBlock(dp)    # 对深度信息上采样

        rgbpre = self.rgbprocess(rgb)  # 对RGB进行3*3卷积
        rgb = self.rgbpre(rgbpre)      # 对RGB信息进行1*1卷积
        spafuse = self.spa_process(torch.cat([dp, rgb], 1))     # 空间信息处理（可逆神经网络），可逆神经网络促进特征融合？
        frefuse = self.fre_process(dp, rgb)   # 频率信息处理SDB

        cat_f = torch.cat([spafuse, frefuse], 1)   # 空间频率联合
        cat_f = self.fuse_process(cat_f)   # 再今行一次可逆神经网络

        cha_res = self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f # （通道维度处理+注意力）* 本身
        out = cha_res + dp  # 与dp加回来

        out = self.downBlock(out)

        return out,rgbpre

class Get_gradient_nopadding_rgb(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding_rgb, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding_d(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding_d, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        x = x0
        return x

class GFE(nn.Module):
    def __init__(self,n_feats,scale):
        super(GFE, self).__init__()
        self.grad_rgb = Get_gradient_nopadding_rgb()
        self.grad_d = Get_gradient_nopadding_d()
        self.upBlock = DenseProjection(1, 1, scale, up=True, bottleneck=False)
        self.downBlock = DenseProjection(n_feats, n_feats, scale, up=False, bottleneck=False)
        self.c_rgb = default_conv(3,n_feats,3)
        self.c_d = default_conv(1,n_feats,3)
        self.c_fuse = default_conv(n_feats,n_feats,3)

        self.rg_d = ResidualGroup(default_conv, n_feats, 3, reduction=16, n_resblocks=4)
        self.rb_rgbd = ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.fuse_process = nn.Sequential(nn.Conv2d(2*n_feats, n_feats, 1, 1, 0),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))
        self.re_g = default_conv(n_feats,1,3)
        self.re_d = default_conv(n_feats,1,3)
        self.c_sab = default_conv(1,n_feats,3)
        self.sig = nn.Sigmoid()
        self.d1 = nn.Sequential(default_conv(1,n_feats,3),
                                ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=8))

        self.CA = CALayer(n_feats,reduction=4)

        grad_conv = [
            default_conv(1, n_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(n_feats, n_feats, kernel_size=3, bias=True),
        ]
        self.grad_conv = nn.Sequential(*grad_conv)
        self.grad_rg = nn.Sequential(ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
        ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))

    def forward(self, depth,rgb):   # 看看与SUFT是否有相同的地方

        depth = self.upBlock(depth)  # 先对深度图上采样

        grad_rgb = self.grad_rgb(rgb)  # 求彩色图梯度
        grad_d = self.grad_d(depth)   # 求深度图梯度

        rgb1 = self.c_rgb(grad_rgb)   # 卷积到c通道，是否加激活？
        d1 = self.c_d(grad_d)

        rgb2 = self.rb_rgbd(rgb1)  # 对彩色图用残差块
        d2 = self.rg_d(d1)       # 对深度图用残差组

        cat1 = torch.cat([rgb2,d2],dim=1)

        inn1 = self.fuse_process(cat1)   # 融合后：选卷积到n_feats，再用两个残差组

        d3 = d1 + self.CA(inn1)   # 通道注意力+残差

        grad_d2 = self.c_fuse(d3) # 残差后的卷积

        out_re = self.re_g(grad_d2)  # 梯度恢复到1通道 c->1

        d4 = self.d1(depth)  # 卷积到n_feats通道 + 残差组

        grad_d3 = self.grad_conv(out_re) + d4  # 1->c，c->c两次卷积， 然后做相加

        grad_d4 = self.grad_rg(grad_d3)  # 最后接上两个残差组

        return out_re,self.downBlock(grad_d4)



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
