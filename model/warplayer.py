import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    device = torch.device("cuda")
    # k = (str(tenFlow.device), str(tenFlow.size())) #该k由光流张量的设备和光流张量的尺寸组成，并转换为字符串
    # if k not in backwarp_tenGrid:   #检查k是否在backwarp_tenGrid中，如果不在，创建一个新的k
        #tenHorizontal是一个从-1到1的线性插值张量，长度为tenFlow的宽度shape[3],并调整形状以匹配光流张量的批量大小和高度
    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)

    #tenVertical 是一个从 -1.0 到 1.0 的线性插值张量，长度为 tenFlow 的高度（shape[2]），
    # 并调整形状以匹配光流张量的批量大小和宽度
    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

    #将tenHorizontal和tenVertival沿着通道维度连接起来，并存储到backwarp_tenGrid[k]中
    backwarp_tenGrid = torch.cat(
        [tenHorizontal, tenVertical], 1).to(device)

        # 对光流张量tenFlow的第一个通道（水平光流）和第二个通道（垂直光流）进行归一化，使其范围与输入图像的宽度和高度相匹配。
        # 然后将这两个通道连接起来形成一个新的光流张量。
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    #将存储在 backwarp_tenGrid 中的网格与归一化后的光流张量相加，得到一个新的网格 g，然后将张量的维度重新排列，以便于后续操作。
    #具体来说，将通道维度（1）移动到最后。
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)

    #它通过双线性插值（或其他指定的插值方法）计算输入张量在采样网格指定位置的值，并生成新的输出图像。
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


