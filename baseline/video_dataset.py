import torch
from torch.utils import data
import numpy as np
import h5py
import os

FEATURES_DIR = '/Users/sripathisridhar/Documents/njit/cs677/project/features_data'


class TAUVideoDataset(data.Dataset):

    def __init__(self, subset='train', features_dir=None):
        super().__init__()
        self.subset = subset
        self.features_dir = features_dir
        if self.features_dir is None:
            self.features_dir = FEATURES_DIR

        # TODO: change to OS agnostic version
        if self.subset == 'train':
            self.path = os.path.join(self.features_dir, 'audio_features_data/tr.hdf5')

        if self.subset == 'val':
            self.path = os.path.join(self.features_dir, 'audio_features_data/val.hdf5')
            #self.path_input = '/lustre/wang9/all_features_data/audio_features_data/cv.hdf5'

        if self.subset == 'test':
            self.path = os.path.join(self.features_dir, 'audio_features_data/tt.hdf5')

        # get means and std deviations to normalize the features
        # global_mean_std_path_audio = os.path.join(self.features_dir,
        #                                           'audio_features_data/global_mean_std.npz')
        # mean_std_audio = np.load(global_mean_std_path_audio)
        # self.mean_audio = mean_std_audio['global_mean']
        # self.std_audio = mean_std_audio['global_std']

        global_mean_std_path_video = os.path.join(self.features_dir,
                                                  'video_features_data/global_mean_std.npz')
        mean_std_video = np.load(global_mean_std_path_video)
        self.mean_video = mean_std_video['global_mean']
        self.std_video = mean_std_video['global_std']

        self.all_files = []
        self.group = []

        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)
            elif isinstance(obj, h5py.Group):
                self.group.append(name)
        self.hf = h5py.File(self.path, 'r')
        self.hf.visititems(func)
        self.hf.close()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):

        # get paths
        # path_audio = self.path
        path_video = self.path.replace('audio', 'video')

        # hf_audio = h5py.File(path_audio, 'r')
        hf_video = h5py.File(path_video, 'r')

        # get normalized embeddings
        # emb_audio = np.array(hf_audio[self.all_files[index]])
        # norm_audio = (emb_audio - self.mean_audio) / self.std_audio

        video_name = self.all_files[index].replace('audio', 'video')
        emb_video = np.array(hf_video[video_name])
        norm_video = (emb_video - self.mean_video) / self.std_video

        # concatenate audio and video embeddings
        # norm_emb = np.concatenate((norm_audio, norm_video))

        # get ground truth from file name
        target = np.array(int(self.all_files[index].split('/')[0]))

        # convert to tensors
        norm_emb_tensor = torch.from_numpy(norm_video).float()
        target_tensor = torch.from_numpy(target).long()

        return norm_emb_tensor, target_tensor
