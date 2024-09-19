<<<<<<< HEAD
import random

import numpy as np
import torch
import torchvision.transforms as T
from torchaudio.functional import add_noise
from torchvision import transforms
from PIL import ImageFilter
import librosa


def do_mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, channel, n_freq_bin, time = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()  # [low, high)

    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1

        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, channel, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + \
                           db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, channel, n_freq_bin, time)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, :, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, :, i].unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, channel, n_freq_band + 1)).to(features) * (
                    db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, channel, n_freq_bin, time)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    for k in range(channel):
                        freq_filt[j, k, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = \
                            torch.linspace(band_factors[j, k, i], band_factors[j, k, i + 1],
                                           band_bndry_freqs[i + 1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)

        return features * freq_filt

    else:
        return features


def FreqMask(mag, num_mask=1, mask_percentage=0.01):
    """
    :param mag: (B, C, H, W) - 输入的张量
    :param num_mask: 掩码的数量
    :param mask_percentage: 掩码的百分比，0.001 ~ 0.015
    """
    B, C, H, W = mag.shape  # 获取张量的维度
    mask_height = int(mask_percentage * H)  # 掩码的高度
    mask = torch.zeros((B, C, H, W), device=mag.device)  # 创建一个全零的掩码张量

    for _ in range(num_mask):  # 对每一个掩码
        # 随机生成掩码的起始位置
        mask_start = torch.randint(0, H - mask_height + 1, (B, C), device=mag.device)
        # 构造掩码张量
        for b in range(B):
            for c in range(C):
                mask[b, c, mask_start[b, c]:mask_start[b, c] + mask_height, :] = 1

    # 应用掩码
    mag = mag * (1 - mask)  # 将掩码应用到原始张量上
    return mag


def filt_aug_prototype(features, db_range=(-7.5, 6), n_bands=(2, 5)):
    # This is FilterAugment algorithm used for DCASE 2021 Challenge Task 4
    batch_size, channels, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_bands[0], high=n_bands[1], size=(1,)).item()  # [low, high)

    if n_freq_band > 1:
        # Generate band boundary frequencies
        band_bndry_freqs = torch.cat((torch.tensor([0]),
                                      torch.sort(torch.randint(1, n_freq_bin - 1, (n_freq_band - 1,)))[0],
                                      torch.tensor([n_freq_bin])))

        # Generate random factors for each band and each channel
        band_factors = torch.rand((batch_size, channels, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + \
                       db_range[0]
        band_factors = 10 ** (band_factors / 20)

        # Initialize the frequency filter with ones
        freq_filt = torch.ones((batch_size, channels, n_freq_bin, 1)).to(features)

        # Apply the filter to each frequency band
        for i in range(n_freq_band):
            freq_filt[:, :, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, :, i].unsqueeze(
                -1).unsqueeze(-1)

        return features * freq_filt
    else:
        return features


def SaltAndPepper(src, percentage):
    B, C, H, W = src.shape  # 获取输入图像的形状
    device = src.device  # 获取输入张量所在的设备
    SP_NoiseImg = src.clone()  # 复制输入图像

    # 计算每个图像的噪声点数目
    SP_NoiseNum = int(percentage * H * W)

    # 随机选择噪声点的索引
    flat_indices = np.random.choice(H * W, SP_NoiseNum, replace=False)

    # 将 NumPy 数组转换为 PyTorch Tensor，并迁移到正确的设备
    flat_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)

    for b in range(B):  # 遍历每个图像
        # 随机生成噪声，并迁移到正确的设备
        noise = torch.randint(0, 2, (SP_NoiseNum,), dtype=torch.float32, device=device)
        # 随机生成通道索引
        randC = torch.randint(0, C, (SP_NoiseNum,), device=device)
        # 计算噪声点的行列索引
        randH = flat_indices // W
        randW = flat_indices % W

        # 将噪声应用到图像
        SP_NoiseImg[b, randC, randH, randW] = noise

    return SP_NoiseImg


class ImageAugmentor:
    def __init__(self,
                 target_size=(224, 224),
                 resize_size=None,
                 crop_size=None,
                 center_crop_size=None,
                 padding=None,
                 flip_prob=0.5,
                 rotation_degree=0,
                 color_jitter_params=None,
                 vertical_flip_prob=0.5,
                 gaussian_blur_radius=0,
                 affine_params=None):
        """
        :param target_size: 最终输出的图像尺寸，格式为 (高度, 宽度)
        :param resize_size: 目标尺寸，格式为 (高度, 宽度)，如果为 None，则不进行缩放
        :param crop_size: 随机裁剪尺寸，格式为 (高度, 宽度)，如果为 None，则不进行随机裁剪
        :param center_crop_size: 中心裁剪尺寸，格式为 (高度, 宽度)，如果为 None，则不进行中心裁剪
        :param padding: 边缘拓展的大小（像素），如果为 None，则不进行边缘拓展
        :param flip_prob: 水平翻转的概率
        :param rotation_degree: 随机旋转的最大角度，如果为 0，则不进行旋转
        :param color_jitter_params: 颜色抖动的参数，格式为 (亮度, 对比度, 饱和度, 色调)，每项为 None 或 (min, max)
        :param vertical_flip_prob: 垂直翻转的概率
        :param gaussian_blur_radius: 高斯模糊的半径，如果为 0，则不进行模糊
        :param affine_params: 仿射变换的参数，格式为 (degrees, translate, scale, shear)，
                              其中 translate 应为 [0, 1] 范围内的值，scale 应为长度为 2 的序列 (min_scale, max_scale)
        """
        self.target_size = target_size
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.center_crop_size = center_crop_size
        self.padding = padding
        self.flip_prob = flip_prob
        self.rotation_degree = rotation_degree
        self.color_jitter_params = color_jitter_params
        self.vertical_flip_prob = vertical_flip_prob
        self.gaussian_blur_radius = gaussian_blur_radius
        self.affine_params = affine_params

        self.transforms = []

        if self.padding is not None:
            self.transforms.append(transforms.Pad(self.padding))

        if self.resize_size is not None:
            self.transforms.append(transforms.Resize(self.resize_size))

        if self.crop_size is not None:
            self.transforms.append(transforms.RandomCrop(self.crop_size))

        if self.center_crop_size is not None:
            self.transforms.append(transforms.CenterCrop(self.center_crop_size))

        if self.flip_prob > 0:
            self.transforms.append(transforms.RandomHorizontalFlip(self.flip_prob))

        if self.vertical_flip_prob > 0:
            self.transforms.append(transforms.RandomVerticalFlip(self.vertical_flip_prob))

        if self.rotation_degree > 0:
            self.transforms.append(transforms.RandomRotation(self.rotation_degree))

        if self.color_jitter_params:
            self.transforms.append(transforms.ColorJitter(*self.color_jitter_params))

        if self.gaussian_blur_radius > 0:
            self.transforms.append(
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(self.gaussian_blur_radius))))

        if self.affine_params:
            degrees, translate, scale, shear = self.affine_params
            translate = (translate[0] / 100, translate[1] / 100)
            if isinstance(scale, (float, int)):
                scale = (scale, scale)
            self.transforms.append(transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear))

        # 最后，确保输出尺寸为 target_size
        self.transforms.append(transforms.Resize(self.target_size))

        self.transform = transforms.Compose(self.transforms)

    def __call__(self, tensor):
        """
        :param tensor: 输入的张量，要求为 (B, C, H, W) 格式
        :return: 数据增强后的张量
        """
        B, C, H, W = tensor.shape

        # 转换为PIL图像进行处理
        tensor_list = [transforms.ToPILImage()(tensor[b]) for b in range(B)]

        # 对每个样本应用变换
        transformed_list = [self.transform(img) for img in tensor_list]

        # 转换回张量格式
        transformed_tensor = torch.stack([transforms.ToTensor()(img) for img in transformed_list])

        return transformed_tensor


def A_weightingAugment(tensor, sample_rate, alpha=1.0, use_mel=True):
    """
    对输入的声谱图进行基于 A-weighting 的增强，支持 STFT 线性频率和 Mel 频率。

    参数:
    tensor: 输入张量，形状为 (B, C, H, W)，其中 H 对应频带数量
    sample_rate: 音频的采样率，用于确定频率分布
    alpha: 控制增强幅度的系数
    use_mel: 布尔值，是否使用 Mel 频率，如果为 False 则使用 STFT 的线性频率

    返回:
    增强后的张量，形状为 (B, C, H, W)
    """
    B, C, H, W = tensor.shape

    if use_mel:
        # 计算 Mel 频率
        frequencies = librosa.mel_frequencies(n_mels=H, fmin=0, fmax=sample_rate / 2)
    else:
        # 计算线性频率 (STFT)
        frequencies = np.linspace(0, sample_rate / 2, H)

    # 使用 np.errstate 来忽略 divide by zero 警告
    with np.errstate(divide='ignore'):
        # 计算 A-weighting 权重
        a_weights = librosa.A_weighting(frequencies)

    # 将 A-weighting 权重转换为 torch 张量
    a_weights = torch.tensor(a_weights, dtype=torch.float32)

    # 将 a_weights 移动到与 tensor 相同的设备上
    a_weights = a_weights.to(tensor.device)

    # 扩展形状以匹配输入张量
    a_weights = a_weights.view(1, 1, H, 1)

    # 对张量进行加权，并使用 alpha 控制增强幅度
    enhanced_tensor = tensor + alpha * tensor * a_weights

    return enhanced_tensor


def spectrogram_random_shifts(tensor, max_shift_time=5, max_shift_freq=5):
    """
    对输入的4D频谱张量进行随机时间和频率平移。

    参数:
    tensor (torch.Tensor): 输入的4D频谱张量，形状为 (B, C, H, W)
    max_shift_time (int): 时间轴上的最大平移量（以时间步为单位）。
    max_shift_freq (int): 频率轴上的最大平移量（以频率步为单位）。

    返回:
    torch.Tensor: 经过随机平移后的4D频谱张量。
    """
    B, C, H, W = tensor.shape

    # 创建一个和输入张量相同大小的零张量
    shifted_tensor = torch.zeros_like(tensor)

    for i in range(B):
        for j in range(C):
            # 随机生成时间和频率的平移量
            shift_time = torch.randint(-max_shift_time, max_shift_time + 1, (1,)).item()
            shift_freq = torch.randint(-max_shift_freq, max_shift_freq + 1, (1,)).item()

            # 计算平移后的索引范围
            time_start = max(0, shift_time)
            time_end = min(W, W + shift_time)

            freq_start = max(0, shift_freq)
            freq_end = min(H, H + shift_freq)

            # 将原始频谱图中对应部分复制到平移后的张量中
            shifted_tensor[i, j, freq_start:freq_end, time_start:time_end] = tensor[
                                                                             i, j,
                                                                             max(0, -shift_freq):min(H, H - shift_freq),
                                                                             max(0, -shift_time):min(W, W - shift_time)
                                                                             ]

    return shifted_tensor


# new methods
def localized_enhancement(spectrogram, region_size=16, gain=1.2):
    B, C, H, W = spectrogram.shape
    for i in range(0, H, region_size):
        for j in range(0, W, region_size):
            region = spectrogram[:, :, i:i + region_size, j:j + region_size]
            # 只对低于0.9的区域进行增益
            enhanced_region = torch.where(region < 0.9, region * gain, region)
            spectrogram[:, :, i:i + region_size, j:j + region_size] = torch.clamp(enhanced_region, 0.0, 1.0)
    return spectrogram


def localized_frequency_band_adjustment(spectrogram, target_band=(50, 150), gain=0.001):
    B, C, H, W = spectrogram.shape
    start_index = int(H * target_band[0] / 224)
    end_index = int(H * target_band[1] / 224)
    spectrogram[:, :, start_index:end_index, :] = spectrogram[:, :, start_index:end_index, :] * gain
    return torch.clamp(spectrogram, 0.0, 1.0)


def random_local_masking(spectrogram, mask_size=16):
    B, C, H, W = spectrogram.shape
    for _ in range(5):
        x_start = torch.randint(0, W - mask_size, (1,)).item()
        y_start = torch.randint(0, H - mask_size, (1,)).item()
        spectrogram[:, :, y_start:y_start + mask_size, x_start:x_start + mask_size] = 0
    return spectrogram


def spatial_transform(spectrogram):
    transform = T.Compose([
        T.RandomRotation(degrees=5),
        T.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0))
    ])
    transformed = transform(spectrogram)
    return torch.clamp(transformed, 0.0, 1.0)


def add_noises(spectrogram, noise_level=0.05):
    noise = torch.randn_like(spectrogram) * noise_level
    noisy_spectrogram = spectrogram + noise
    # 将值裁剪到 [0, 1] 范围
    return torch.clamp(noisy_spectrogram, 0.0, 1.0)


def hybrid_augmentation(spectrogram):
    spectrogram = localized_frequency_band_adjustment(spectrogram, target_band=[30, 50], gain=1.2)
    spectrogram = random_local_masking(spectrogram, mask_size=20)
    spectrogram = spatial_transform(spectrogram)
    spectrogram = add_noises(spectrogram)  # 添加轻微噪声
    return spectrogram


def nonlinear_frequency_transformation(spectrogram, transform_type='log', factor=1.0):
    if transform_type == 'log':
        transformed = torch.log1p(spectrogram * factor) / torch.log1p(torch.tensor(1.0 * factor))
    elif transform_type == 'exp':
        transformed = torch.expm1(spectrogram * factor) / torch.expm1(torch.tensor(1.0 * factor))
    else:
        raise ValueError("Unsupported transform type.")
    return torch.clamp(transformed, 0.0, 1.0)


def diverse_frequency_masking(spectrogram, num_masks=3, max_mask_pct=0.15):
    B, C, H, W = spectrogram.shape
    for _ in range(num_masks):
        mask_width = int(max_mask_pct * H)
        f_start = torch.randint(0, H - mask_width, (1,)).item()
        spectrogram[:, :, f_start:f_start + mask_width, :] = 0
    return spectrogram


def dynamic_background_simulation(spectrogram, noise_spectrogram=None, dynamic_factor=0.00001,
                                  noise_level_range=(0, 0.00001)):
    """
    对输入的频谱图应用动态背景噪音增强，且噪音强度在一定范围内随机变化。

    参数:
    spectrogram (torch.Tensor): 输入的频谱图，形状为 (B, C, H, W)。
    noise_spectrogram (torch.Tensor, optional): 用于叠加的背景噪音频谱图，形状为 (B, C, H, W)。
                                                如果未提供，则生成随机噪音频谱图。
    dynamic_factor (float): 控制动态变化的因子。
    noise_level_range (tuple): 噪音强度的范围，随机选择一个值作为当前噪音的强度。

    返回:
    torch.Tensor: 经过动态背景噪音增强后的频谱图，形状为 (B, C, H, W)。
    """
    device = spectrogram.device
    B, C, H, W = spectrogram.shape

    # 在噪音强度范围内随机选择一个值
    noise_level = torch.FloatTensor(1).uniform_(*noise_level_range).item()

    if noise_spectrogram is None:
        noise_spectrogram = torch.randn(B, C, H, W, device=device) * noise_level

    dynamic_wave = torch.sin(torch.linspace(0, 2 * torch.pi, W, device=device)) * dynamic_factor + 1
    dynamic_noise = noise_spectrogram * dynamic_wave.unsqueeze(0).unsqueeze(0)

    return spectrogram + dynamic_noise


def get_band(x, min_band_size, max_band_size, band_type, mask):
    assert band_type.lower() in ['freq', 'time'], f"band_type must be in ['freq', 'time']"
    if band_type.lower() == 'freq':
        axis = 2  # 频率轴
    else:
        axis = 3  # 时间轴
    band_size = random.randint(min_band_size, max_band_size)
    mask_start = random.randint(0, x.size()[axis] - band_size)
    mask_end = mask_start + band_size

    if band_type.lower() == 'freq':
        mask[:, :, mask_start:mask_end, :] = 1
    if band_type.lower() == 'time':
        mask[:, :, :, mask_start:mask_end] = 1
    return mask

def specmix(x, y, prob, min_band_size, max_band_size, max_frequency_bands=3, max_time_bands=3):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    device = x.device  # 获取x的设备

    k = torch.rand(1, device=device)
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size, device=device)
        mask = torch.zeros((x.size()[0], x.size()[1], x.size()[2], x.size()[3]), device=device)
        num_frequency_bands = random.randint(1, max_frequency_bands)
        for i in range(1, num_frequency_bands):
            mask = get_band(x, min_band_size, max_band_size, 'freq', mask)
        num_time_bands = random.randint(1, max_time_bands)
        for i in range(1, num_time_bands):
            mask = get_band(x, min_band_size, max_band_size, 'time', mask)
        lam = torch.sum(mask) / (x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3])
        x = x * (1 - mask) + x[batch_idx] * mask
        y = y * (1 - lam) + y[batch_idx] * lam
        return x, y
    else:
        return x, y
=======
import random

import numpy as np
import torch
import torchvision.transforms as T
from torchaudio.functional import add_noise
from torchvision import transforms
from PIL import ImageFilter
import librosa


def do_mixup(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def specmix(x, y, prob, min_band_size, max_band_size, max_frequency_bands=3, max_time_bands=3):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    device = x.device

    k = torch.rand(1, device=device)
    if k > 1 - prob:
        batch_size = x.size()[0]
        batch_idx = torch.randperm(batch_size, device=device)
        mask = torch.zeros((x.size()[0], x.size()[1], x.size()[2], x.size()[3]), device=device)
        num_frequency_bands = random.randint(1, max_frequency_bands)
        for i in range(1, num_frequency_bands):
            mask = get_band(x, min_band_size, max_band_size, 'freq', mask)
        num_time_bands = random.randint(1, max_time_bands)
        for i in range(1, num_time_bands):
            mask = get_band(x, min_band_size, max_band_size, 'time', mask)
        lam = torch.sum(mask) / (x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3])
        x = x * (1 - mask) + x[batch_idx] * mask
        y = y * (1 - lam) + y[batch_idx] * lam
        return x, y
    else:
        return x, y
>>>>>>> 9706377d6644a0a4fab142b1f67327a5bc8acac2
