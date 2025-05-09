# Import necessary libraries
# PyTorch for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter  # Add TensorBoard support

# Data processing and visualization
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from IPython.display import display
from tqdm import tqdm  # For progress bars
import time  # For timing training steps

def load_data(base_path):
    """
    Load medical image data from numpy arrays.
    
    Args:
        base_path (str): Base directory path containing the data files
        
    Returns:
        tuple: (image_array, types_array, masks_array)
            - image_array: Medical images (N, H, W, 3)
            - types_array: Image type labels
            - masks_array: Segmentation masks (N, H, W, 6)
            
    Raises:
        AssertionError: If data dimensions don't match expected format
        Exception: If files can't be loaded or processed
    """
    try:
        # Use os.path.join for cross-platform compatibility
        images_path = os.path.join(base_path, "Part 1", "Images", "images.npy")
        types_path = os.path.join(base_path, "Part 1", "Images", "types.npy")
        masks_path = os.path.join(base_path, "Part 1", "Masks", "masks.npy")
        
        print(f"Looking for files at:")
        print(f"Images: {images_path}")
        print(f"Types: {types_path}")
        print(f"Masks: {masks_path}")
        
        # Load numpy arrays
        image_array = np.load(images_path)
        types_array = np.load(types_path)
        masks_array = np.load(masks_path)
        
        # Validate shapes and dimensions
        assert len(image_array.shape) == 4, "Images should be 4D (N, H, W, C)"
        assert image_array.shape[-1] == 3, "Images should have 3 channels (RGB)"
        assert image_array.shape[0] == types_array.shape[0] == masks_array.shape[0], \
            "Number of samples mismatch between arrays"
        assert masks_array.shape[-1] == 6, "Expected 6 classes in masks"
        
        return image_array, types_array, masks_array
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Load data with error handling
try:
    base_path = os.path.join(os.path.dirname(__file__), "archive")
    print(f"\nBase path: {base_path}")
    image_array, types_array, masks_array = load_data(base_path)
except Exception as e:
    print(f"Failed to load data: {e}")
    exit(1)

# Normalize pixel values to range [0,1]
def normalize_images(images):
    """
    Normalize image pixel values to range [0,1].
    
    Args:
        images (np.ndarray): Input images with pixel values in range [0,255]
        
    Returns:
        np.ndarray: Normalized images with pixel values in range [0,1]
    """
    return images / 255.0

# Convert segmentation masks to one-hot encoding
# This creates a binary mask for each class
def one_hot_encode_masks(masks, num_classes=6):
    """
    Convert segmentation masks to one-hot encoding.
    
    Args:
        masks (np.ndarray): Input masks with class indices
        num_classes (int): Number of classes in the segmentation task
        
    Returns:
        np.ndarray: One-hot encoded masks with shape (N, H, W, num_classes)
    """
    # Initialize array for one-hot encoded masks
    one_hot = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], num_classes))
    # For each class, create a binary mask
    for i in range(num_classes):
        one_hot[..., i] = (masks[..., i] > 0).astype(np.float32)
    return one_hot

# Define image transformations
# Normalize using ImageNet statistics as we're using a pre-trained model
image_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
    transforms.ConvertImageDtype(torch.float32),  # Ensure float32
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Define mask transformations - only convert to tensor, no normalization
class ToFloatTensor:
    """
    Custom transform to convert numpy arrays to float32 tensors.
    This is used for masks to ensure proper data type without normalization.
    """
    def __call__(self, mask):
        return torch.from_numpy(mask).float()

mask_transform = transforms.Compose([
    ToFloatTensor()
])

# Custom PyTorch Dataset class for medical images
class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for medical images and their segmentation masks.
    
    Attributes:
        images (np.ndarray): Input medical images
        masks (np.ndarray): Segmentation masks
        image_transform (callable): Transformations for images
        mask_transform (callable): Transformations for masks
    """
    def __init__(self, images, masks, image_transform=None, mask_transform=None):
        self.images = images
        self.masks = masks
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and corresponding mask
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Apply transformations if specified
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        # Ensure mask is in the correct format (C, H, W)
        if mask.dim() == 3 and mask.shape[0] != 6:  # If channels are last dimension
            mask = mask.permute(2, 0, 1)
            
        return image, mask

# Add memory management
def preprocess_data(image_array, types_array, masks_array):
    """
    Preprocess the data and handle memory management.
    
    Args:
        image_array (np.ndarray): Raw image data
        types_array (np.ndarray): Image type labels
        masks_array (np.ndarray): Raw mask data
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            Preprocessed and split data for training, validation, and testing
    """
    try:
        # Normalize images and encode masks
        normalized_images = normalize_images(image_array)
        one_hot_masks = one_hot_encode_masks(masks_array)
        
        # Split data into train/val/test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            normalized_images, one_hot_masks, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Clean up memory
        del image_array, types_array, masks_array, normalized_images, one_hot_masks, X_temp, y_temp
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

# Add validation function for data types
def validate_data_types(dataset, num_samples=5):
    """
    Validate data types and shapes of images and masks in the dataset.
    
    Args:
        dataset (MedicalImageDataset): Dataset to validate
        num_samples (int): Number of samples to check
        
    Raises:
        ValueError: If data types or shapes are incorrect
    """
    print("\nValidating data types and shapes:")
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"Image - Type: {image.dtype}, Shape: {image.shape}")
        print(f"Mask - Type: {mask.dtype}, Shape: {mask.shape}")
        
        # Verify data types and shapes
        if image.dtype != torch.float32:
            raise ValueError(f"Image should be float32, got {image.dtype}")
        if mask.dtype != torch.float32:
            raise ValueError(f"Mask should be float32, got {mask.dtype}")
        if mask.shape[0] != 6:
            raise ValueError(f"Mask should have 6 channels, got {mask.shape[0]}")

# Preprocess the data to get training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(image_array, types_array, masks_array)

# Create PyTorch datasets with separate transforms
train_dataset = MedicalImageDataset(X_train, y_train, 
                                  image_transform=image_transform,
                                  mask_transform=mask_transform)
val_dataset = MedicalImageDataset(X_val, y_val,
                                image_transform=image_transform,
                                mask_transform=mask_transform)
test_dataset = MedicalImageDataset(X_test, y_test,
                                 image_transform=image_transform,
                                 mask_transform=mask_transform)

# Validate data types before training
print("Validating training data...")
validate_data_types(train_dataset)
print("\nValidating validation data...")
validate_data_types(val_dataset)
print("\nValidating test data...")
validate_data_types(test_dataset)

# Create data loaders for batch processing
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model for semantic segmentation.
    
    Architecture:
    1. ResNet50 backbone for feature extraction
    2. ASPP module for multi-scale context
    3. Decoder for upsampling and final prediction
    
    Attributes:
        expected_channels (dict): Expected number of channels at each stage
    """
    def __init__(self, num_classes=6):
        super(DeepLabV3Plus, self).__init__()
        
        # Load pre-trained ResNet-50 as backbone
        backbone = models.resnet50(pretrained=True)
        
        # Extract specific layers for feature extraction
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels
        
        # ASPP module for multi-scale feature extraction
        self.aspp = ASPP(2048, 256)
        
        # Low-level feature processing
        self.low_level_conv = nn.Conv2d(256, 48, 1)
        
        # Decoder for upsampling and final prediction
        self.decoder = Decoder(304, 256, num_classes)
        
        # Store expected channel dimensions for validation
        self.expected_channels = {
            'initial_out': 64,
            'layer1_out': 256,
            'layer2_out': 512,
            'layer3_out': 1024,
            'layer4_out': 2048,
            'aspp_out': 256,
            'low_level_out': 48,
            'decoder_in': 304,
            'decoder_out': num_classes
        }
        
    def validate_channels(self, tensor, stage_name):
        """
        Validate channel dimensions at each stage of the network.
        
        Args:
            tensor (torch.Tensor): Tensor to validate
            stage_name (str): Name of the network stage
            
        Raises:
            ValueError: If channel dimensions don't match expected values
        """
        expected_channels = self.expected_channels[stage_name]
        actual_channels = tensor.size(1)
        if actual_channels != expected_channels:
            raise ValueError(f"Channel dimension mismatch at {stage_name}. "
                           f"Expected {expected_channels}, got {actual_channels}")
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, H, W)
            
        Raises:
            ValueError: If input size is too small or channel dimensions don't match
        """
        # Validate input size
        if x.size(2) < 32 or x.size(3) < 32:
            raise ValueError("Input size too small for ResNet50. Minimum size is 32x32")
        
        # Extract features through ResNet50
        x = self.initial(x)
        self.validate_channels(x, 'initial_out')
        
        low_level_features = self.layer1(x)  # 256 channels
        self.validate_channels(low_level_features, 'layer1_out')
        
        x = self.layer2(low_level_features)
        self.validate_channels(x, 'layer2_out')
        
        x = self.layer3(x)
        self.validate_channels(x, 'layer3_out')
        
        x = self.layer4(x)
        self.validate_channels(x, 'layer4_out')
        
        # Process through ASPP for multi-scale context
        x = self.aspp(x)
        self.validate_channels(x, 'aspp_out')
        
        # Process low-level features
        low_level_features = self.low_level_conv(low_level_features)
        self.validate_channels(low_level_features, 'low_level_out')
        
        # Upsample ASPP output to match low-level feature size
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate features for better detail preservation
        x = torch.cat([x, low_level_features], dim=1)
        self.validate_channels(x, 'decoder_in')
        
        # Final prediction through decoder
        x = self.decoder(x)
        self.validate_channels(x, 'decoder_out')
        
        return x

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    This module captures multi-scale context by employing multiple parallel
    atrous convolutions with different rates.
    
    Attributes:
        conv1 (nn.Sequential): 1x1 convolution branch
        conv2-4 (nn.Sequential): 3x3 convolutions with different dilation rates
        global_avg_pool (nn.Sequential): Global average pooling branch
        conv_final (nn.Sequential): Final 1x1 convolution
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # 1x1 convolution branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Three 3x3 convolutions with different dilation rates
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution to combine all branches
        self.conv_final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass through the ASPP module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Processed tensor with multi-scale context
        """
        # Process through all branches
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        # Upsample global average pooling output
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate all branches
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        # Final processing
        x = self.conv_final(x)
        
        return x

class Decoder(nn.Module):
    """
    Decoder module for upsampling and final prediction.
    
    This module processes the concatenated features from ASPP and low-level
    features to produce the final segmentation map.
    
    Attributes:
        conv1-2 (nn.Sequential): Convolution blocks for feature processing
        conv3 (nn.Conv2d): Final 1x1 convolution for class prediction
    """
    def __init__(self, in_channels, mid_channels, num_classes):
        super(Decoder, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution for class prediction
        self.conv3 = nn.Conv2d(mid_channels, num_classes, 1)
        
    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor from ASPP and low-level features
            
        Returns:
            torch.Tensor: Final segmentation map
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def calculate_f1_score(preds, targets, num_classes=6):
    """
    Calculate F1 score for each class and return mean F1 score.
    
    Args:
        preds (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        num_classes (int): Number of classes
        
    Returns:
        tuple: (mean_f1, f1_scores)
            - mean_f1: Mean F1 score across all classes
            - f1_scores: F1 scores for each class
    """
    f1_scores = []
    
    # Convert predictions to class labels
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_mask = (preds == class_idx)
        target_mask = (targets == class_idx)
        
        # Calculate true positives, false positives, and false negatives
        tp = torch.sum(pred_mask & target_mask).float()
        fp = torch.sum(pred_mask & ~target_mask).float()
        fn = torch.sum(~pred_mask & target_mask).float()
        
        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)
    
    # Convert to numpy array and calculate mean
    f1_scores = torch.stack(f1_scores)
    mean_f1 = torch.mean(f1_scores)
    
    return mean_f1, f1_scores

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (avg_loss, avg_f1, avg_class_f1)
            - avg_loss: Average loss across the dataset
            - avg_f1: Mean F1 score
            - avg_class_f1: F1 scores for each class
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_f1 = 0.0
    class_f1_scores = torch.zeros(6, device=device)
    
    with torch.no_grad():  # Disable gradient computation
        for images, masks in dataloader:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Convert one-hot encoded masks to class indices
            masks_indices = torch.argmax(masks, dim=1)
            
            # Forward pass
            outputs = model(images)
            # Upsample outputs to match target size
            outputs = F.interpolate(outputs, size=masks_indices.shape[1:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, masks_indices)
            
            # Calculate F1 score
            mean_f1, class_f1 = calculate_f1_score(outputs, masks)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_f1 += mean_f1.item()
            class_f1_scores += class_f1
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    avg_class_f1 = class_f1_scores / len(dataloader)
    
    return avg_loss, avg_f1, avg_class_f1

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, 
                early_stopping_patience=5, max_grad_norm=1.0, accumulation_steps=4):
    """
    Train the model with gradient accumulation and mixed precision.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        early_stopping_patience (int): Number of epochs to wait for improvement
        max_grad_norm (float): Maximum gradient norm for clipping
        accumulation_steps (int): Number of steps to accumulate gradients
        
    Returns:
        tuple: (trained_model, history)
            - trained_model: The trained model
            - history: Dictionary containing training metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize best validation F1 score and early stopping counter
    best_val_f1 = 0.0
    early_stopping_counter = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'class_f1': []
    }
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Convert one-hot encoded masks to class indices
            masks_indices = torch.argmax(masks, dim=1)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                # Upsample outputs to match target size
                outputs = F.interpolate(outputs, size=masks_indices.shape[1:], mode='bilinear', align_corners=True)
                loss = criterion(outputs, masks_indices)
                loss = loss / accumulation_steps  # Normalize loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        val_loss, val_f1, class_f1 = evaluate_model(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['class_f1'].append(class_f1.cpu().numpy())
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Print metrics
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Loss - Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}')
        print(f'F1 Score - Mean: {val_f1:.4f}')
        print('Class-wise F1 Scores:')
        for i, f1 in enumerate(class_f1):
            print(f'  Class {i}: {f1:.4f}')
        print('-' * 50)
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'class_f1': class_f1,
            }, 'best_model.pth')
            print(f'New best model saved with F1 score: {val_f1:.4f}')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Dictionary containing training metrics
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Mean F1')
    for i in range(6):
        class_f1 = [epoch[i] for epoch in history['class_f1']]
        plt.plot(class_f1, label=f'Class {i} F1')
    plt.title('Validation F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Add visualization function
def visualize_results(model, dataloader, device, num_samples=4):
    """
    Visualize model predictions on sample images.
    
    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader for visualization
        device (torch.device): Device to run inference on
        num_samples (int): Number of samples to visualize
    """
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            
            # Convert predictions to class labels
            preds = torch.argmax(preds, dim=1)
            masks = torch.argmax(masks, dim=1)
            
            # Plot original image
            axes[i, 0].imshow(images[0].cpu().permute(1, 2, 0))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Plot ground truth mask
            axes[i, 1].imshow(masks[0].cpu(), cmap='jet')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Plot predicted mask
            axes[i, 2].imshow(preds[0].cpu(), cmap='jet')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Add class weights for handling class imbalance
def calculate_class_weights(masks):
    """
    Calculate class weights based on inverse frequency.
    
    Args:
        masks (np.ndarray): One-hot encoded masks
        
    Returns:
        torch.Tensor: Class weights for handling class imbalance
    """
    class_counts = np.sum(masks, axis=(0, 1, 2))
    total_pixels = np.prod(masks.shape[:3])
    class_weights = total_pixels / (class_counts + 1e-8)  # Add epsilon to avoid division by zero
    return torch.FloatTensor(class_weights)

# Add a test function to verify channel dimensions
def test_channel_dimensions():
    """Test function to verify channel dimensions throughout the network"""
    model = DeepLabV3Plus(num_classes=6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create a dummy input tensor (batch_size=1, channels=3, height=256, width=256)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    try:
        # Forward pass with validation
        output = model(dummy_input)
        print("Channel dimension validation passed successfully!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except ValueError as e:
        print(f"Channel dimension validation failed: {e}")
        raise

# Main training block
try:
    print("\n=== Starting Model Training Process ===")
    
    # Load and preprocess data
    print("\n1. Loading data...")
    base_path = os.path.join(os.path.dirname(__file__), "archive")
    image_array, types_array, masks_array = load_data(base_path)
    print("   Data loaded successfully!")
    
    print("\n2. Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        image_array, types_array, masks_array
    )
    print("   Data preprocessing completed!")
    
    # Create datasets and dataloaders
    print("\n3. Creating datasets...")
    train_dataset = MedicalImageDataset(X_train, y_train, 
                                      image_transform=image_transform,
                                      mask_transform=mask_transform)
    val_dataset = MedicalImageDataset(X_val, y_val,
                                    image_transform=image_transform,
                                    mask_transform=mask_transform)
    test_dataset = MedicalImageDataset(X_test, y_test,
                                     image_transform=image_transform,
                                     mask_transform=mask_transform)
    print("   Datasets created successfully!")
    
    # Validate data types
    print("\n4. Validating data types...")
    print("   Validating training data...")
    validate_data_types(train_dataset)
    print("   Validating validation data...")
    validate_data_types(val_dataset)
    print("   Validating test data...")
    validate_data_types(test_dataset)
    print("   Data validation completed!")
    
    # Create model and training components
    print("\n5. Initializing model and training components...")
    model = DeepLabV3Plus(num_classes=6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    model = model.to(device)
    
    # Calculate class weights
    print("\n6. Calculating class weights...")
    class_weights = calculate_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("   Training components initialized!")
    
    # Train the model with gradient accumulation and mixed precision
    print("\n7. Starting model training...")
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=10, early_stopping_patience=5,
        accumulation_steps=4  # Accumulate gradients for 4 batches
    )
    print("   Model training completed!")
    
    # Plot training history
    print("\n8. Plotting training history...")
    plot_training_history(history)
    print("   Training history plotted!")
    
    # Evaluate on test set
    print("\n9. Evaluating model on test set...")
    test_loss, test_f1, test_class_f1 = evaluate_model(trained_model, test_loader, criterion, device)
    print(f'\nTest Results:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Mean F1 Score: {test_f1:.4f}')
    print('Class-wise F1 Scores:')
    for i, f1 in enumerate(test_class_f1):
        print(f'  Class {i}: {f1:.4f}')
    
    # Visualize results
    print("\n10. Visualizing results...")
    visualize_results(trained_model, test_loader, device)
    print("   Results visualization completed!")
    
    print("\n=== Model Training Process Completed Successfully! ===")
    
except Exception as e:
    print(f"\nError during model training: {e}")
    raise
