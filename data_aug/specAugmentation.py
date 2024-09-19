<<<<<<< HEAD
import librosa
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.

        Args:
          dim: int, dimension along which to drop (2 for freq, 3 for time)
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: frequency; dim 3: time

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, freq_bins, time_steps)"""

        assert input.ndimension() == 4

        if not self.training:
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
            distance = torch.randint(low=0, high=self.drop_width, size=(1,)).item()
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,)).item()

            if self.dim == 2:  # Drop along the frequency axis
                e[:, bgn:bgn + distance, :] = -40.0 / 255
            elif self.dim == 3:  # Drop along the time axis
                e[:, :, bgn:bgn + distance] = -40.0 / 255


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width,
                 freq_stripes_num):
        """Spec augmentation.

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=3, drop_width=time_drop_width,
                                        stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=2, drop_width=freq_drop_width,
                                        stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


if __name__ == '__main__':
    matrix = np.load('D:\WMWB\stft\Acrocephalus arundinaceus\XC417157_1.npy')
    input = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0)
    spec_aug = SpecAugmentation(time_drop_width=10, time_stripes_num=2,
                                freq_drop_width=20, freq_stripes_num=2)
    output = spec_aug(input)
    output = output.squeeze(0).squeeze(0).numpy()
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
import torch.nn as nn
from matplotlib import pyplot as plt

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        super(DropStripes, self).__init__()
        assert dim in [2, 3]
        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        assert input.ndimension() == 4
        if not self.training:
            return input
        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]
            for n in range(batch_size):
                self.transform_slice(input[n], total_width)
            return input

    def transform_slice(self, e, total_width):
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,)).item()
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,)).item()

            if self.dim == 2:
                e[:, bgn:bgn + distance, :] = -40.0 / 255
            elif self.dim == 3:
                e[:, :, bgn:bgn + distance] = -40.0 / 255

class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num):
        super(SpecAugmentation, self).__init__()
        self.time_dropper = DropStripes(dim=3, drop_width=time_drop_width, stripes_num=time_stripes_num)
        self.freq_dropper = DropStripes(dim=2, drop_width=freq_drop_width, stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x
>>>>>>> 9706377d6644a0a4fab142b1f67327a5bc8acac2
