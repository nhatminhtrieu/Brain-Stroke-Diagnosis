#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Standard library imports
import os
import zipfile
from collections import Counter

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Torchvision imports
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


# In[2]:


if torch.cuda.is_available():
    print(f"GPUs Available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"- {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")


# In[3]:


DATASET_NAME = 'rsna-mil-training'
HPC_DIR = '/media02/tdhoang01/21127112-21127734/data'

ZIP_FILE_PATH = os.path.join(HPC_DIR, DATASET_NAME + '.zip')  # Path to the zip file

DICOM_DIR = f'{DATASET_NAME}/'

# Paths within the zip, using DATASET_NAME
CSV_PATH = f'{DATASET_NAME}/training_1000_scan_subset.csv'
SLICE_LABEL_PATH = 'sorted_training_dataset_with_labels.csv'

# Load CSVs from zip
with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
    patient_scan_labels = pd.read_csv(zip_ref.open(CSV_PATH))
    patient_slice_labels = pd.read_csv(zip_ref.open(SLICE_LABEL_PATH))


# In[4]:


MAX_SLICES = 60

HEIGHT = 224
WIDTH = 224

BATCH_PATIENTS = 8

VAL_SIZE = 0.15
TEST_SIZE = 0.15

TARGET_COLUMNS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


# In[5]:


patient_scan_labels.head(1)


# In[6]:


patient_slice_labels.head(1)


# In[8]:


class DatasetGenerator(Dataset):
    def __init__(self, zip_file_path, patient_scan_labels, patient_slice_labels, max_slices, height, width, target_columns):
        self.zip_file_path = zip_file_path
        self.patient_scan_labels = patient_scan_labels
        self.patient_slice_labels = patient_slice_labels
        self.max_slices = max_slices
        self.height = height
        self.width = width
        self.target_columns = target_columns
        self.channels = 3
        self.dicom_paths = self._get_dicom_paths()
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width))
        ])

    def __len__(self):
        return len(self.dicom_paths)

    def __getitem__(self, idx):
        dicom_files, _ = self.dicom_paths[idx]
        images, labels = self._process_dicom_files(dicom_files)
        return self._pad_data(images, labels)

    def _get_dicom_paths(self):
        dicom_paths = []
        with zipfile.ZipFile(self.zip_file_path, 'r') as dicom_zip:
            zip_file_list = dicom_zip.namelist()
            top_level_folder = f"{DATASET_NAME}/"

            for _, row in self.patient_scan_labels.iterrows():
                patient_id = row['patient_id'].replace("ID_", "")
                study_instance_uid = row['study_instance_uid'].replace("ID_", "")
                dicom_dir_path = f"{top_level_folder}{patient_id}_{study_instance_uid}/"
                dicom_files = [f for f in zip_file_list if f.startswith(dicom_dir_path) and f.endswith(".dcm")]
                
                if dicom_files:
                    dicom_paths.append((dicom_files, row))
                else:
                    print(f"No DICOM files found in {dicom_dir_path} within the zip file.")
        
        return dicom_paths

    def _process_dicom_files(self, dicom_files):
        images = []
        labels = []
        with zipfile.ZipFile(self.zip_file_path, 'r') as dicom_zip:
            for dicom_file in dicom_files:
                with dicom_zip.open(dicom_file) as file:
                    dicom = pydicom.dcmread(file)
                    img = self._preprocess_slice(dicom)
                    images.append(torch.from_numpy(img).float())
                    labels.append(self._get_label(dicom_file))
        
        return torch.stack(images), torch.tensor(labels, dtype=torch.float32)

    def _preprocess_slice(self, dicom):
        bsb_img = self._bsb_window(dicom)
        return bsb_img.astype(np.float16)

    def _get_label(self, dicom_file):
        file_key = os.path.basename(dicom_file)
        label_row = self.patient_slice_labels[self.patient_slice_labels['filename'] == file_key]
        return 1.0 if not label_row.empty and np.any(label_row[self.target_columns].values == 1) else 0.0

    def _pad_data(self, images, labels):
        if images.shape[0] < self.max_slices:
            padding = torch.zeros((self.max_slices - images.shape[0], self.channels, self.height, self.width))
            images = torch.cat((images, padding), dim=0)
            label_padding = torch.zeros(self.max_slices - labels.shape[0])
            labels = torch.cat((labels, label_padding))
        return images, labels

    def _correct_dcm(self, dcm):
        x = dcm.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] -= px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000

    def _window_image(self, dcm, window_center, window_width):
        if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
            self._correct_dcm(dcm)
        
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
       
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        
        return img

    def _bsb_window(self, dcm):
        brain_img = self._window_image(dcm, 40, 80)
        subdural_img = self._window_image(dcm, 80, 200)
        soft_img = self._window_image(dcm, 40, 380)
        
        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        soft_img = (soft_img - (-150)) / 380
        
        bsb_img = np.stack([brain_img, subdural_img, soft_img], axis=0)
        return bsb_img


# In[9]:


class DataloaderManager:
    def __init__(self, dataset, batch_size, val_size, test_size, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.train_loader = None
        self.validate_loader = None
        self.test_loader = None
        self._create_loaders()

    def print_dataset_length(self):
        print(f"Length of dataset: {len(self.dataset)}")

    def print_lengths(self):
        print(f"Length of training dataset: {len(self.train_loader.dataset)}")
        print(f"Length of validation dataset: {len(self.validate_loader.dataset)}")
        print(f"Length of testing dataset: {len(self.test_loader.dataset)}")
    
    def _create_loaders(self):
        dataset_length = len(self.dataset)
        print(f"Dataset length: {dataset_length}")

        # Create masks for indices with and without label 1
        label_mask = np.array([1 in sample[1].tolist() for sample in self.dataset])
        indices_with_one = np.where(label_mask)[0]
        indices_without_one = np.where(~label_mask)[0]

        print(f"Total indices with one: {len(indices_with_one)}")

        # Calculate split sizes
        val_size = int(dataset_length * self.val_size)
        test_size = int(dataset_length * self.test_size)

        # Shuffle and split indices
        np.random.shuffle(indices_with_one)
        np.random.shuffle(indices_without_one)

        val_one_count = int(len(indices_with_one) * self.val_size)
        test_one_count = int(len(indices_with_one) * self.test_size)

        val_indices = np.concatenate((
            indices_with_one[:val_one_count],
            indices_without_one[:val_size - val_one_count]
        ))

        test_indices = np.concatenate((
            indices_with_one[val_one_count:val_one_count + test_one_count],
            indices_without_one[val_size - val_one_count:val_size - val_one_count + test_size - test_one_count]
        ))

        train_indices = np.concatenate((
            indices_with_one[val_one_count + test_one_count:],
            indices_without_one[val_size - val_one_count + test_size - test_one_count:]
        ))

        # Create datasets from selected indices using Subset
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        test_dataset = Subset(self.dataset, test_indices)

        # Create DataLoaders for each set
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.validate_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def visualize_label_distribution(self):
        for loader_name, loader in zip(['Training', 'Validation', 'Testing'], 
                                        [self.train_loader, self.validate_loader, self.test_loader]):
            all_labels = []
            for _, labels in loader:
                all_labels.extend(labels.view(-1).numpy().tolist())

            label_counts = Counter(all_labels)
            counts = [label_counts.get(0, 0), label_counts.get(1, 0)]
            labels = [0, 1]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, counts, color='blue')
            plt.xlabel('Labels')
            plt.ylabel('Counts')
            plt.title(f'Label Distribution - {loader_name} Set')
            plt.xticks(labels)

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

            plt.tight_layout()
            plt.show()


# In[10]:


dataset = DatasetGenerator(
    zip_file_path=ZIP_FILE_PATH,
    patient_scan_labels=patient_scan_labels,
    patient_slice_labels=patient_slice_labels,
    max_slices=MAX_SLICES,
    height=HEIGHT,
    width=WIDTH,
    target_columns=TARGET_COLUMNS
)


# In[11]:


images, labels = dataset[0]

print(images.shape, labels.shape)


# In[12]:


dataloader_manager = DataloaderManager(dataset, batch_size=BATCH_PATIENTS, val_size=VAL_SIZE, test_size=TEST_SIZE)

# Accessing the loaders
train_loader = dataloader_manager.train_loader
validate_loader = dataloader_manager.validate_loader
test_loader = dataloader_manager.test_loader


# In[13]:


dataloader_manager.print_lengths()


# In[14]:


dataloader_manager.visualize_label_distribution()


# In[15]:


def plot_image_grid(dataloader, grid_rows=10, max_slices=MAX_SLICES):
    """
    Plot a grid of images from the given DataLoader.
    
    Args:
    dataloader (DataLoader): DataLoader containing batches of images and labels.
    grid_rows (int): Number of rows in the grid
    max_slices (int): Maximum number of slices to plot
    """
    # Get a first value in batch from the DataLoader
    for batch_images, batch_labels in dataloader:
        image_tensor = batch_images[0]
        label_tensor = batch_labels[0]
        break

    num_slices = min(image_tensor.shape[0], max_slices)  # Ensure we don't exceed the number of slices
    grid_cols = int(max_slices / grid_rows)  # Calculate number of columns
    
    # Calculate the figure size based on the image dimensions
    img_size = HEIGHT # or WIDTH
    dpi = plt.rcParams['figure.dpi']  # Get the default DPI
    figsize = (grid_cols * img_size / dpi, grid_rows * img_size / dpi)
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    # Loop through the number of slices and plot each image
    for i in range(num_slices):
        axes[i].imshow(image_tensor[i].permute(1, 2, 0).cpu().numpy())  # Convert to (height, width, channels) for plotting
        axes[i].set_title(f"Label: {label_tensor[i].item()}")
        axes[i].axis('off')  # Turn off axis for all subplots
    
    # Turn off any remaining empty subplots
    for i in range(num_slices, grid_rows * grid_cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# In[16]:


# plot_image_grid(test_loader)


# In[17]:


# 2. Model Definition
class ResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.dropout = nn.Dropout(p=0.3)  # Add dropout layer

    def forward(self, x):
        batch_patients, num_slices, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        x = self.resnet(x)
        return self.dropout(x)  # Apply dropout before returning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)


# In[18]:


# 3. Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# mode: Monitors 'min' or 'max' changes in metrics.
# factor: The multiplicative factor for reducing the learning rate. 
##  If the current learning rate is 0.01 and factor=0.5, the new learning rate will be 0.01 * 0.5 = 0.005
# patience: Number of epochs to wait for improvement before reducing the learning rate.


# In[19]:


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(image).squeeze()
        label = label.reshape(-1)

        loss = criterion(output, label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Calculate accuracy
        predicted_label = (output > 0.5).float()
        total += label.size(0)
        correct += (predicted_label == label).sum().item()

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}%')

    scheduler.step(epoch_loss)


# In[20]:


def validate(model, validate_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(validate_loader):
            image, label = image.to(device), label.to(device)
            
            # Forward pass
            output = model(image).squeeze()
            label = label.reshape(-1)
            
            loss = criterion(output, label)
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted_label = (output > 0.5).float()
            total += label.size(0)
            correct += (predicted_label == label).sum().item()
    
    # Calculate average loss and accuracy for validation
    val_loss = running_loss / len(validate_loader)
    accuracy = 100 * correct / total
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}%')


# In[21]:


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


# In[22]:


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


# In[23]:


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            
            output = model(image).squeeze()
            label = label.reshape(-1)
            
            predicted_label = (output > 0.5).float()
            total += label.size(0)
            correct += (predicted_label == label).sum().item()
            
            # Collecting true labels and scores for ROC and confusion matrix
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_label.cpu().numpy())
            all_scores.extend(output.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.4f}%')
    
    # Plot ROC Curve
    plot_roc_curve(np.array(all_labels), np.array(all_scores))
    
    # Plot Confusion Matrix
    plot_confusion_matrix(np.array(all_labels), np.array(all_predictions))


# In[24]:


num_epochs = 10


# In[ ]:


for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, scheduler, device)
    validate(model, validate_loader, criterion, device)


# In[27]:


evaluate(model, test_loader, device)

