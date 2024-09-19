<<<<<<< HEAD
import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class RandomReplacementDropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num, replacement='noise'):
        """Randomly replace stripes.

        Args:
          dim: int, dimension along which to drop (2 for freq, 3 for time)
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
          replacement: str, 'noise' or 'random'
        """
        super(RandomReplacementDropStripes, self).__init__()

        assert dim in [2, 3]
        assert replacement in ['noise', 'random']

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num
        self.replacement = replacement

    def forward(self, input):
        """input: (batch_size, channels, freq_bins, time_steps)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input

    def transform_slice(self, e, total_width):
        """e: (channels, freq_bins, time_steps)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                if self.replacement == 'noise':
                    e[:, bgn: bgn + distance, :] = torch.clamp(torch.randn_like(e[:, bgn: bgn + distance, :]) * 20 - 60,
                                                               min=-80, max=0) / 255
                elif self.replacement == 'random':
                    e[:, bgn: bgn + distance, :] = torch.rand_like(e[:, bgn: bgn + distance, :])
            elif self.dim == 3:
                if self.replacement == 'noise':
                    e[:, :, bgn: bgn + distance] = torch.clamp(torch.randn_like(e[:, :, bgn: bgn + distance]) * 20 - 60,
                                                               min=-80, max=0) / 255
                elif self.replacement == 'random':
                    e[:, :, bgn: bgn + distance] = torch.rand_like(e[:, :, bgn: bgn + distance])


class RandomReplacementSpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width,
                 freq_stripes_num, replacement='noise'):
        """Random replacement Spec augmentation.

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
          replacement: str, 'noise' or 'random'
        """

        super(RandomReplacementSpecAugmentation, self).__init__()

        self.time_dropper = RandomReplacementDropStripes(dim=3, drop_width=time_drop_width,
                                                         stripes_num=time_stripes_num,
                                                         replacement=replacement)

        self.freq_dropper = RandomReplacementDropStripes(dim=2, drop_width=freq_drop_width,
                                                         stripes_num=freq_stripes_num,
                                                         replacement=replacement)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


if __name__ == '__main__':
    matrix = np.load('D:/WMWB/stft/Acrocephalus arundinaceus/XC417157_1.npy')
    # matrix = matrix / 255

    # Convert to tensor and add channel dimensions
    matrix = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0)

    # Apply augmentation
    aug = RandomReplacementSpecAugmentation(time_drop_width=10, time_stripes_num=2, freq_drop_width=20,
                                            freq_stripes_num=2, replacement='noise')
    output = aug(matrix)

    # Remove the added dimensions and convert back to numpy
    output = output.squeeze(0).squeeze(0).numpy()

    # Plot using librosa
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(output, sr=None, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram after Augmentation')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
=======
import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class RandomReplacementDropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num, replacement='noise'):
        super(RandomReplacementDropStripes, self).__init__()
        assert dim in [2, 3]
        assert replacement in ['noise', 'random']
        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num
        self.replacement = replacement

    def forward(self, input):
        assert input.ndimension() == 4
        if self.training is False:
            return input
        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)
            return input

    def transform_slice(self, e, total_width):
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]
            if self.dim == 2:
                if self.replacement == 'noise':
                    e[:, bgn: bgn + distance, :] = torch.clamp(torch.randn_like(e[:, bgn: bgn + distance, :]) * 20 - 60, min=-80, max=0) / 255
                elif self.replacement == 'random':
                    e[:, bgn: bgn + distance, :] = torch.rand_like(e[:, bgn: bgn + distance, :])
            elif self.dim == 3:
                if self.replacement == 'noise':
                    e[:, :, bgn: bgn + distance] = torch.clamp(torch.randn_like(e[:, :, bgn: bgn + distance]) * 20 - 60, min=-80, max=0) / 255
                elif self.replacement == 'random':
                    e[:, :, bgn: bgn + distance] = torch.rand_like(e[:, :, bgn: bgn + distance])

class RandomReplacementSpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num, replacement='noise'):
        super(RandomReplacementSpecAugmentation, self).__init__()
        self.time_dropper = RandomReplacementDropStripes(dim=3, drop_width=time_drop_width, stripes_num=time_stripes_num, replacement=replacement)
        self.freq_dropper = RandomReplacementDropStripes(dim=2, drop_width=freq_drop_width, stripes_num=freq_stripes_num, replacement=replacement)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x
>>>>>>> 9706377d6644a0a4fab142b1f67327a5bc8acac2
