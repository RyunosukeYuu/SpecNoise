import math
import os
import argparse
import json

import cv2
import librosa
import numpy as np
import random

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class BirdSpectrogramLoader:
    def __init__(self, input_path, test=0.2, verbose=False, is_multi_feature=False, random_seed=None):
        self.input = input_path
        self.test = test
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.is_multi_feature = is_multi_feature
        self.random_seed = random_seed

        if self.test < 0 or self.test > 1:
            raise ValueError('Error, test must be a float between 0 and 1')

        self.train_split = round(1 - self.test, 2)
        if self.train_split < 0:
            raise ValueError('Error, test can\'t add to more than 1')

        print("Input split: train {}%, test {}%".format(self.train_split * 100, self.test * 100))
        if self.verbose:
            print("===== Dataset =====")

    def __split(self):
        cuts_per_file = {}
        for dir in os.listdir(self.input):
            folder_path = os.path.join(self.input, dir)

            if not os.path.isdir(folder_path):
                continue

            cuts_per_file[dir] = []
            for file_name in os.listdir(folder_path):
                cuts_per_file[dir].append(file_name)

        train_dict = {}
        test_dict = {}
        for d, files in cuts_per_file.items():
            random.shuffle(files)
            total_files = random.sample(files, 200)
            num_total = len(total_files)
            num_train = int(self.train_split * num_total)
            num_test = num_total - num_train

            train_dict[d] = total_files[:num_train]
            test_dict[d] = total_files[num_train:num_train + num_test]

        return train_dict, test_dict

    def __get_all_images(self, old_dict):
        updated_dict = {}
        for label, arr in old_dict.items():
            updated_dict[label] = []
            image_path = os.path.join(self.input, label)
            for file_name in os.listdir(image_path):
                if file_name in arr:
                    updated_dict[label].append(file_name)
        return updated_dict

    def __load(self, image_dict, verbose=-1):
        data = []
        labels = []

        for label, arr in image_dict.items():
            i = 0
            label_path = os.path.join(self.input, label)
            for image_name in arr:
                image_path = os.path.join(label_path, image_name)
                image = np.load(image_path) / 255.0
                # Expand to three channels
                if not self.is_multi_feature:
                    image = np.expand_dims(image, axis=-1)
                    image = np.repeat(image, 3, axis=-1)

                image = np.transpose(image, (2, 0, 1))
                data.append(image)
                labels.append(label)
                i += 1

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("\r[INFO] processed {}/{} {}\t\t\t".format(i + 1, len(arr), label), end='')

        print('')
        aux_list = list(zip(data, labels))
        random.shuffle(aux_list)
        data, labels = zip(*aux_list)
        labels = self.label_encoder.fit_transform(np.array(labels))

        return np.array(data), labels

    def load_and_split(self):
        print("Splitting...")
        self.train_dict, self.test_dict = self.__split()

        print("Getting all names for train...")
        self.train_dict = self.__get_all_images(self.train_dict)
        print("Getting all names for test...")
        self.test_dict = self.__get_all_images(self.test_dict)

        print("Loading train images...")
        X_train, Y_train = self.__load(self.train_dict, verbose=100)
        print("Loading test images...")
        X_test, Y_test = self.__load(self.test_dict, verbose=50)
        return X_train, Y_train, X_test, Y_test


class BirdSoundLoader:
    def __init__(self, input_path, test=0.2, verbose=False, random_seed=None, feature_type='None'):
        self.input = input_path
        self.test = test
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.random_seed = random_seed
        self.feature_type = feature_type  # feature_type：'stft', 'mel', 'mfcc', 'all_feature', or 'none'

        if self.test < 0 or self.test > 1:
            raise ValueError('Error, test must be a float between 0 and 1')

        self.train_split = 1 - self.test
        if self.train_split < 0:
            raise ValueError('Error, test cannot be greater than 1')

        print("Input split: train {}%, test {}%".format(self.train_split * 100, self.test * 100))
        if self.verbose:
            print("===== Dataset =====")

    def __split(self):
        cuts_per_file = {}
        for dir in os.listdir(self.input):
            folder_path = os.path.join(self.input, dir)

            if not os.path.isdir(folder_path):
                continue

            cuts_per_file[dir] = []
            for file_name in os.listdir(folder_path):
                cuts_per_file[dir].append(file_name)

        train_dict = {}
        test_dict = {}
        for d, files in cuts_per_file.items():
            random.shuffle(files)
            total_files = files[:200]
            num_train = int(self.train_split * 200)
            num_test = 200 - num_train

            train_dict[d] = total_files[:num_train]
            test_dict[d] = total_files[num_train:num_train + num_test]

        return train_dict, test_dict

    def __get_all_images(self, old_dict):
        updated_dict = {}
        for label, arr in old_dict.items():
            updated_dict[label] = []
            image_path = os.path.join(self.input, label)
            for file_name in os.listdir(image_path):
                if file_name in arr:
                    updated_dict[label].append(file_name)
        return updated_dict

    def __extract_features(self, y, time_split=1.0, n_mels=224):
        hop_length = time_split * 22050 / n_mels
        hop_length = math.ceil(hop_length)

        if self.feature_type == 'stft':
            stft_spec = librosa.stft(y=y, n_fft=512, hop_length=hop_length)
            stft_spec = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max)
            stft_spec = cv2.resize(stft_spec, (224, 224))
            stft_spec = np.repeat(stft_spec[np.newaxis, :, :], 3, axis=0)
            return stft_spec

        if self.feature_type == 'mel':
            mel_spec = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=n_mels, hop_length=hop_length)
            mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
            mel_spec = cv2.resize(mel_spec, (224, 224))
            mel_spec = np.repeat(mel_spec[np.newaxis, :, :], 3, axis=0)
            return mel_spec

        if self.feature_type == 'mfcc':
            mfcc_spec = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=20, hop_length=hop_length)
            mfcc_spec = librosa.util.normalize(mfcc_spec)
            mfcc_spec = cv2.resize(mfcc_spec, (224, 224))
            mfcc_spec = np.repeat(mfcc_spec[np.newaxis, :, :], 3, axis=0)
            return mfcc_spec

        if self.feature_type == 'all_feature':
            stft_spec = librosa.stft(y=y, n_fft=512, hop_length=hop_length)
            stft_spec = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max)
            stft_spec = cv2.resize(stft_spec, (224, 224))

            mel_spec = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=n_mels, hop_length=hop_length)
            mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
            mel_spec = cv2.resize(mel_spec, (224, 224))

            mfcc_spec = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=20, hop_length=hop_length)
            mfcc_spec = librosa.util.normalize(mfcc_spec)
            mfcc_spec = cv2.resize(mfcc_spec, (224, 224))

            combined_features = np.stack([stft_spec, mel_spec, mfcc_spec], axis=0)

            return combined_features

    def __load(self, file_dict, verbose=-1):
        data = []
        labels = []

        for label, arr in file_dict.items():
            i = 0
            label_path = os.path.join(self.input, label)
            for file_name in arr:
                file_path = os.path.join(label_path, file_name)

                y = np.load(file_path)

                if self.feature_type == 'none':
                    y = np.expand_dims(y, axis=0)
                    data.append(y)
                else:
                    # 特征提取
                    features = self.__extract_features(y)
                    data.append(features)

                labels.append(label)
                i += 1

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("\r[INFO] processed {}/{} {}\t\t\t".format(i + 1, len(arr), label), end='')

        print('')
        aux_list = list(zip(data, labels))

        if self.random_seed is not None:
            random.seed(self.random_seed)
        random.shuffle(aux_list)

        data, labels = zip(*aux_list)
        labels = self.label_encoder.fit_transform(np.array(labels))

        return np.array(data), labels

    def load_and_split(self):
        print("Splitting...")
        self.train_dict, self.test_dict = self.__split()

        print("Getting all names for train...")
        self.train_dict = self.__get_all_images(self.train_dict)
        print("Getting all names for test...")
        self.test_dict = self.__get_all_images(self.test_dict)

        print("Loading train data...")
        X_train, Y_train = self.__load(self.train_dict, verbose=100)
        if self.test > 0:
            print("Loading test data...")
            X_test, Y_test = self.__load(self.test_dict, verbose=20)
            return X_train, Y_train, X_test, Y_test

        return X_train, Y_train
