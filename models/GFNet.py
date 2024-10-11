from models.common import *
import torch
import torch.nn as nn
from utils import make_coord





class GFNet(nn.Module):
    def __init__(self, num_feats, kernel_size, scale):
        super(GFNet, self).__init__()
        self.num_layer=3
        self.conv_rgb1 = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_rb3 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rgb_rb4 = ResBlock(default_conv, num_feats, kernel_size, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.conv_dp2 = nn.Conv2d(in_channels=num_feats, out_channels=2*num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)  # 残差组数量 4—>6
        self.dp_rg2 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 2*num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats,scale=scale)
        self.bridge2 = SDM(channels=2*num_feats, rgb_channels=num_feats,scale=scale)
        self.bridge3 = SDM(channels=3*num_feats, rgb_channels=num_feats,scale=scale)

        self.c_de = default_conv(4*num_feats, 2*num_feats, 1)

        my_tail = [                   # mytail也多了一个
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(
                default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=8),
            # ResidualGroup(
            #     default_conv, 3*num_feats, kernel_size, reduction=16, n_resblocks=8)
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(3*num_feats, 3*num_feats, scale, up=True, bottleneck=False)  # 128—>96
        last_conv = [
            default_conv(3*num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, num_feats, kernel_size=3, bias=True)
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.c_rd = default_conv(8*num_feats, 3*num_feats, 1)        # 多了这几个卷积
        self.c_grad = default_conv(2*num_feats, num_feats, 1)
        self.c_grad2 = default_conv(3*num_feats, 2*num_feats, 1)
        self.c_grad3 = default_conv(3*num_feats, 3*num_feats, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.gradNet = GCM(n_feats=num_feats,scale=scale)    # 多了求梯度的GCM

        hidden_dim = num_feats
        # self.imnet = MLP(imnet_in_dim, out_dim=2, hidden_list=[1024, 512, 256, 128])

        layers = []
        # for i in range(2):
        #     layers.append(ResBlock(default_conv, hidden_dim, kernel_size, bias=True, bn=False,
        #                         act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1))
        #
        # layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True))

        # hidden_dim = hidden_dim + self.encoder.dim * 2
        # 接下来尝试将 1*1 卷积都改成3*3
        for i in range(self.num_layer):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True))
        self.layers = nn.Sequential(*layers)

        self.conv_du1 = nn.Conv2d(hidden_dim, num_feats, kernel_size=3, padding=1)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.decoder = nn.Conv2d(num_feats, 1, kernel_size=3, padding=1)
        self.layers = nn.Sequential(*layers)

    def query(self, feat, coord, hr_guide, lr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr
        # B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1), mode='nearest', align_corners=False)
        # q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
        #              :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []
        areas = []
        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, :, 0] += (vx) * rx
                coord_[:, :, :, 1] += (vy) * ry
                k += 1

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                # q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                #          :].permute(0, 2, 1)  # [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                # q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                #           :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False) # [B, N, 2]

                rel_coord = coord.permute(0, 3, 1, 2) - q_coord
                rel_coord[:, 0, :, :] *= h
                rel_coord[:, 1, :, :] *= w

                # q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                #              :, :, 0, :].permute(0, 2, 1)  # [B, N, C]
                # q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)
                inp = q_feat * q_guide_hr
                # inp = torch.cat([q_feat, q_guide_hr], dim=1)

                inp_1 = self.layers(inp)
                inp = inp + inp_1
                inp = self.act1(self.conv_du1(inp))

                # pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(inp)

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1)
        # preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        # weight = F.softmax(preds[:, 1, :, :, :], dim=-1)
        #
        # ret = (preds[:, 0, :, :, :] * weight).sum(-1, keepdim=True)

        return ret


    def forward(self, x):
        image, depth, coord, cell = x
        b, c, h, w = image.shape
        _, _, hl, wl = depth.shape

        out_re, grad_d4 = self.gradNet(depth, image)  # 对原始图像求梯度

        dp_in = self.act(self.conv_dp1(depth))
        dp1 = self.dp_rg1(dp_in)

        cat10 = torch.cat([dp1, grad_d4], dim=1)
        dp1_ = self.c_grad(cat10)

        rgb1 = self.act(self.conv_rgb1(image))
        rgb2 = self.rgb_rb2(rgb1)

        ca1_in, r1 = self.bridge1(dp1_, rgb2)
        dp2 = self.dp_rg2(torch.cat([dp1, ca1_in + dp_in], 1))

        cat11 = torch.cat([dp2, grad_d4], dim=1)
        dp2_ = self.c_grad2(cat11)

        rgb3 = self.rgb_rb3(r1)
        ca2_in, r2 = self.bridge2(dp2_, rgb3)

        ca2_in_ = ca2_in + self.conv_dp2(dp_in)

        cat1_0 = torch.cat([dp2, ca2_in_], 1)

        dp3 = self.dp_rg3(self.c_de(cat1_0))
        rgb4 = self.rgb_rb4(r2)

        cat12 = torch.cat([dp3, grad_d4], dim=1)
        dp3_ = self.c_grad3(cat12)

        ca3_in, r3 = self.bridge3(dp3_, rgb4)

        cat1 = torch.cat([dp1, dp2, dp3, ca3_in], 1)

        dp4 = self.last_conv(self.dp_rg4(self.c_rd(cat1)))

        r3_l = F.interpolate(r3, size=(hl, wl), mode='bicubic')
        res = self.query(dp4, coord, r3, r3_l)  # 只是将hr_image cancat到通道里

        res = self.decoder(res)
        res = res + self.bicubic(depth)
        # tail_in = self.upsampler(dp4)
        # out = self.last_conv(self.tail(tail_in))
        # out = out + self.bicubic(depth)

        return res, out_re

    # def query(self, feat, coord, hr_guide, lr_guide):
#
#     # feat: [B, C, h, w]
#     # coord: [B, N, 2], N <= H * W
#
#     b, c, h, w = feat.shape  # lr
#     B, N, _ = coord.shape
#
#     # LR centers' coords
#     feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
#
#     q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
#                  :].permute(0, 2, 1)  # [B, N, C]
#
#     rx = 1 / h
#     ry = 1 / w
#
#     preds = []
#
#     k = 0
#     for vx in [-1, 1]:
#         for vy in [-1, 1]:
#             coord_ = coord.clone()
#
#             coord_[:, :, 0] += (vx) * rx
#             coord_[:, :, 1] += (vy) * ry
#             k += 1
#
#             # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
#             q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
#                      :].permute(0, 2, 1)  # [B, N, c]
#             q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
#                       :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]
#
#             rel_coord = coord - q_coord
#             rel_coord[:, :, 0] *= h
#             rel_coord[:, :, 1] *= w
#
#             q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
#                          :, :, 0, :].permute(0, 2, 1)  # [B, N, C]
#             q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)
#
#             inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)
#
#             pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
#             preds.append(pred)
#
#     preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
#     weight = F.softmax(preds[:, :, 1, :], dim=-1)
#
#     ret = (preds[:, :, 0, :] * weight).sum(-1, keepdim=True)
#
#     return ret
