import torch
import numpy as np

from utils import data_process

class DongNaiDataset:
    def __init__(self, data_dir, patient_scan_labels, augmentor=None):
        self.data_dir = data_dir
        self.dataset = self._parse_patient_scan_labels(patient_scan_labels)
        self.augmentor = augmentor

    def _parse_patient_scan_labels(self, patient_scan_labels):
        # Convert patient_id to string and patient_label to boolean
        patient_scan_labels['patient_id'] = patient_scan_labels['patient_id'].astype(str)
        patient_scan_labels['patient_label'] = patient_scan_labels['patient_label'].astype(bool)
        return patient_scan_labels

    def _process_patient_data(self, row):
        return data_process.process_patient_data(self.data_dir, row, dataset='dongnai')

    def __len__(self):
        return len(self.dataset) * (self.augmentor.levels if self.augmentor else 1)

    def __getitem__(self, idx):
        patient_idx = idx // (self.augmentor.levels if self.augmentor else 1)
        aug_level = idx % (self.augmentor.levels if self.augmentor else 1)

        row = self.dataset.iloc[patient_idx]
        preprocessed_slices, labels = self._process_patient_data(row)

        preprocessed_slices = self._prepare_tensor(preprocessed_slices, aug_level if self.augmentor else None)
        patient_label = torch.tensor(bool(row['patient_label']), dtype=torch.uint8)
        multi_class_labels = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.uint8) # We don't have multi-class labels for Dong Nai dataset

        return preprocessed_slices, labels, patient_label, multi_class_labels

    def _prepare_tensor(self, preprocessed_slices, aug_level):
        # Convert to numpy array and then to torch tensor
        preprocessed_slices = np.asarray(preprocessed_slices, dtype=np.float32)
        preprocessed_slices = torch.tensor(preprocessed_slices, dtype=torch.float32)

        # Add an additional dimension for channel if it's missing (no augmentor)
        if preprocessed_slices.ndim == 3:
            preprocessed_slices = preprocessed_slices.unsqueeze(1)  # shape: [slices, 1, H, W]

        # Apply augmentation if augmentor is specified
        if self.augmentor and aug_level is not None:
            if preprocessed_slices.ndim == 4:  # Ensure it has the [slices, channels, H, W] format
                return torch.stack([self.augmentor.apply_transform(img, aug_level) for img in preprocessed_slices])

        return preprocessed_slices  # Return without augmentation if augmentor is None
