# For GAN based few-shot method
"""
CelebA Dataloader implementation, used in DCGAN
"""
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from utils.preprocess import *

# For GAN based few-shot method
class FewShot_Dataloader(Dataset):
    # Returns support and query images for few-shot learning
    def __init__(self, config, phase):
        self.config = config
        assert phase in ['training', 'validating', 'testing']
        self.phase = phase
        
        if phase == 'training':
            # Labeled data (support)
            self.patches, self.label = preprocess_dynamic_lab(
                config.data_directory, 
                config.seed, 
                config.num_classes,
                config.extraction_step, 
                config.patch_shape,
                config.number_images_training,
                config.num_images_validating,
                config.num_images_testing,
            )

            # Unlabeled data (query)
            self.patches_unlab = preprocess_dynamic_unlab(
                config.data_directory, 
                config.extraction_step,
                config.patch_shape, 
                config.number_unlab_images_training
            )

            self.labeled_count = len(self.patches)
            self.unlabeled_count = len(self.patches_unlab)

            self.episodes_count = self.config.episodes_per_epoch

            # Unlabeled data (query) - pseudo-labels
            self.unlab_labels = np.zeros((self.unlabeled_count, *self.label.shape[1:]), dtype=self.label.dtype)

        if phase == 'validating':
            self.patches, self.label, self.whole_vol = preprocess_dynamic_lab(
                config.data_directory,
                config.seed,
                config.num_classes,
                config.extraction_step,
                config.patch_shape,
                config.number_images_training,
                config.num_images_validating,
                config.num_images_testing,
                validating=True
            )

        if phase == 'testing':
            self.patches, self.whole_vol = preprocess_dynamic_lab(
                config.data_directory,
                config.seed,
                config.num_classes,
                config.extraction_step,
                config.patch_shape,
                config.number_images_training,
                config.num_images_validating,
                config.num_images_testing,
                testing=True
            )

    def __len__(self):
        return self.episodes_count if self.phase == 'training' else len(self.patches)

    def __getitem__(self, index):
        if self.phase == 'training':
            # Select 5 random support images from labeled data
            s_indices = np.random.choice(self.labeled_count, size=5, replace=False)
            support_imgs = self.patches[s_indices]      # (5, C, D, H, W)
            support_labels = self.label[s_indices]      # (5, D, H, W)

            # Select 1 random query image from unlabeled data
            q_idx = np.random.randint(0, self.unlabeled_count)
            query_img = self.patches_unlab[q_idx]       # (C, D, H, W)
            query_label = self.unlab_labels[q_idx]      # (D, H, W), pseudo-label (0)

            support_imgs_t = torch.from_numpy(support_imgs).float()
            support_labels_t = torch.from_numpy(support_labels).long()
            query_img_t = torch.from_numpy(query_img).float()
            query_label_t = torch.from_numpy(query_label).long()

            return support_imgs_t, support_labels_t, query_img_t, query_label_t

        elif self.phase == 'validating':
            patch_t = torch.from_numpy(self.patches[index]).float()
            label_t = torch.from_numpy(self.label[index]).long()
            return patch_t, label_t, self.whole_vol

        elif self.phase == 'testing':
            patch_t = torch.from_numpy(self.patches[index]).float()
            return patch_t, self.whole_vol

class FewShot_Dataset:
    def __init__(self, config, phase):
        self.config = config
        self.dataset = FewShot_Dataloader(config, phase)
        
        if phase == 'training':
            shuffle = True
        else:
            shuffle = False
            
        self.loader = DataLoader(self.dataset,
                                 batch_size=config.batch_size,
                                 shuffle=shuffle,
                                 num_workers=config.data_loader_workers,
                                 pin_memory=config.pin_memory)

        self.num_iterations = len(self.loader)

    def finalize(self):
        pass
