import os
import torch
import pydicom
import numpy as np
import cv2
from skimage.transform import resize
import sys

import yaml
if os.path.exists('/media02/tdhoang01/python-debugging/config.yaml'):
    path = '/media02/tdhoang01/python-debugging/config.yaml'
elif os.path.exists('/workspace/Brain-Stroke-Diagnosis/config.yaml'):
    path = '/workspace/Brain-Stroke-Diagnosis/config.yaml'
else:
    path = '../config.yaml'
with open(path) as file:
    config = yaml.safe_load(file)

# Accessing constants from config
HEIGHT = config['height']
WIDTH = config['width']
CHANNELS = config['channels']

TRAIN_BATCH_SIZE = config['train_batch_size']
VALID_BATCH_SIZE = config['valid_batch_size']
TEST_BATCH_SIZE = config['test_batch_size']
TEST_SIZE = config['test_size']
VALID_SIZE = config['valid_size']

MAX_SLICES = config['max_slices']
SHAPE = tuple(config['shape'])

NUM_EPOCHS = config['num_epochs']
LEARNING_RATE = config['learning_rate']
INDUCING_POINTS = config['inducing_points']
THRESHOLD = config['threshold']

NUM_CLASSES = config['num_classes']

TARGET_LABELS = config['target_labels']

MODEL_PATH = config['model_path']
DEVICE = config['device']


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def create_bone_mask(dcm):
    # Assuming dcm.pixel_array contains the HU values
    hu_values = dcm.pixel_array

    # Create a mask for bone regions
    # bone_mask = (hu_values >= 1000) & (hu_values <= 1200)
    bone_mask = (hu_values >= 1000) & (hu_values <= 1200)
    return bone_mask


def extract_bone_mask(dcm):
    # Create the bone mask
    bone_mask = create_bone_mask(dcm)

    # Extract the bone mask from the image
    hu_values = dcm.pixel_array.copy()
    # hu_values[bone_mask] = 0
    hu_values[~bone_mask] = 0

    # Update the DICOM pixel data
    dcm.PixelData = hu_values.tobytes()


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    # extract_bone_mask(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    # Resize
    img = cv2.resize(img, SHAPE[:2], interpolation=cv2.INTER_LINEAR)

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    # soft_img = window_image(dcm, 40, 380)
    soft_img = window_image(dcm, 30, 30)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    # soft_img = (soft_img - (-150)) / 380
    soft_img = (soft_img - 15) / 30

    if CHANNELS == 3:
        bsb_img = np.stack([brain_img, subdural_img, soft_img], axis=-1)
    else:
        bsb_img = brain_img
        # weight = [0.4, 0.3, 0.3]
        # # Create weighted grayscale composite
        # bsb_img = (brain_img * weight[0] +
        #            subdural_img * weight[1] +
        #            soft_img * weight[2])
        # # Maintain channel dimension for consistency
        # bsb_img = np.expand_dims(bsb_img, axis=-1)

    return bsb_img.astype(np.float16)


def preprocess_slice(slice, target_size=(HEIGHT, WIDTH)):
    # Check if type of slice is dicom or an empty numpy array
    if (type(slice) == np.ndarray):
        slice = resize(slice, target_size, anti_aliasing=True)
        multichannel_slice = np.stack([slice, slice, slice], axis=-1)
        if CHANNELS == 3:
            return multichannel_slice.astype(np.float16)
        else:
            return slice.astype(np.float16)
    else:
        slice = bsb_window(slice)
        return slice


def read_dicom_files(folder_path, filenames, max_slices=MAX_SLICES):
    try:
        # Read and sort DICOM files based on ImagePositionPatient
        dicom_files = sorted(
            [os.path.join(folder_path, f) for f in filenames if f.endswith(".dcm")],
            key=lambda f: float(pydicom.dcmread(f).ImagePositionPatient[2])
        )[:max_slices]

        # Read and store slices
        slices = [pydicom.dcmread(f) for f in dicom_files]

        # Pad with black images if necessary
        if len(slices) < max_slices:
            black_image = np.zeros_like(slices[0].pixel_array)
            slices += [black_image] * (max_slices - len(slices))

    except:
        print(f"Error reading DICOM files in {folder_path}")
        return []

    return slices[:max_slices]

def read_dicom_files_cq500(folder_path, filenames, max_slices=32):
    try:
        # Read and sort DICOM files based on ImagePositionPatient
        dicom_files = sorted(
            [os.path.join(folder_path, f) for f in filenames if f.endswith(".dcm")],
            key=lambda f: float(pydicom.dcmread(f).ImagePositionPatient[2])
        )[:max_slices]

        # Read and store slices
        slices = [pydicom.dcmread(f) for f in dicom_files]

        # Pad with black images if necessary
        if len(slices) < max_slices:
            black_image = np.zeros_like(slices[0].pixel_array)
            slices += [black_image] * (max_slices - len(slices))

        return slices[:max_slices]
    # If catch an error, return an empty list
    except:
        print(f"Error reading DICOM files in {folder_path}")
        return []

    return slices[:max_slices]


def process_patient_data(dicom_dir, row, num_instances=12, depth=5, dataset='rsna'):
    if dataset == 'rsna':
        patient_id = row['patient_id'].replace('ID_', '')
        study_instance_uid = row['study_instance_uid'].replace('ID_', '')

        folder_name = f"{patient_id}_{study_instance_uid}"
        folder_path = os.path.join(dicom_dir, folder_name)
    else: 
        folder_name = row['name']
        folder_path = os.path.join('./archive', folder_name, 'Unknown Study', row['Source Folder'])

    if os.path.exists(folder_path):
        # Get the filenames from the row
        filenames = row['filename']

        if dataset == 'rsna':
            # Read only the specified DICOM files
            slices = read_dicom_files(folder_path, filenames)
        else: 
            # Read all DICOM files in the folder
            slices = read_dicom_files_cq500(folder_path, filenames)

        # Preprocess slices and convert to tensor
        # preprocessed_slices = [torch.tensor(preprocess_slice(slice), dtype=torch.float32) for slice in slices]
        preprocessed_slices = []
        for slice in slices:
            slice = torch.tensor(preprocess_slice(slice), dtype=torch.float32)
            if slice.ndim == 2:
                slice = slice.unsqueeze(-1)
            preprocessed_slices.append(slice)
        # Stack preprocessed slices into an array
        preprocessed_slices = torch.stack(preprocessed_slices, dim=0)  # (num_slices, height, width, channels)

        padded_labels = torch.zeros(len(preprocessed_slices), dtype=torch.long)

        return preprocessed_slices, padded_labels

    else:
        print(f"Folder not found: {folder_name}")
        sys.exit(1)