import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf #pip install tensorflow[and-cuda]
import seaborn as sns

from tensorflow.keras.models import Model #pip install keras[and-cuda]
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.metrics import MeanIoU
 
image_size=1024

def plot_history(history, save_path):
    """
    
    Description:

    This function plots the model's training history and saves it

    Parameters:

    history - model's training history

    save_path - path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(save_path)
    plt.show()


def unet_model(input_size):
    """
    
    Description:

    This function creates U-net model

    Parameters:

    input_size - size of input image (height, width, number of bands)
    """
    inputs = Input(input_size)
    # Downsampling
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)

    # Upsampling
    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(256, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(128, 3, activation='relu', padding='same')(c7)

    u8 = UpSampling2D()(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(64, 3, activation='relu', padding='same')(c8)

    u9 = UpSampling2D()(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(32, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Data generators to manage memory usage
def data_generator(image_dataset, mask_dataset, batch_size):
    """

    Description:

    This function creates data generator

    Parameters:

    image_dataset - image dataset for a model

    mask_dataset - mask dataset for a model

    batch_size - size of batch in a model
    """
    while True:
        for i in range(0, len(image_dataset), batch_size):
            img_batch = image_dataset[i:i+batch_size]
            mask_batch = mask_dataset[i:i+batch_size]
            yield img_batch, mask_batch

def get_model_accuarcy(ground_truth, predictions):
    true_masks=[]
    predicted_masks=[]

    for true_mask, predicted_mask in zip(ground_truth, predictions):
        true_mask_flat=true_mask.flatten()
        predicted_mask_flat=(predicted_mask.flatten() > 0.5).astype(int)

        true_masks.extend(true_mask_flat)
        predicted_masks.extend(predicted_mask_flat)

    true_masks=np.array(true_masks)
    predicted_masks=np.array(predicted_masks)

    producer_accuracy = recall_score(true_masks, predicted_masks, average=None)
    user_accuracy = precision_score(true_masks, predicted_masks, average=None)
    results = {}
    #0 - non oil, 1 - oil
    for i, (producer, user) in enumerate(zip(producer_accuracy, user_accuracy)):
        print(f"Class {i}: Producer's Accuracy (Recall) = {producer:.2f}, User's Accuracy (Precision) = {user:.2f}")
        results[f'Class {i}'] = {'Producer Accuracy (Recall)': producer, 'User Accuracy (Precision)': user}

    return results


def calculate_metrics(true_masks, predicted_masks, threshold=0.5):
    """
    Calculate performance metrics for U-Net model on segmentation task.

    Parameters:
    - true_masks (np.array): Ground truth masks (binary)
    - predicted_masks (np.array): Predicted masks (raw output)
    - threshold (float): Threshold for binarizing predicted masks (default=0.5)

    Returns:
    - metrics (dict): Dictionary containing accuracy, precision, recall, F1-score, IoU, and Dice coefficient.
    """
    
    # Binarize predicted masks
    predicted_masks_bin = (predicted_masks > threshold).astype(np.uint8)
    
    # Flatten the masks for metric calculation
    true_masks_flat = true_masks.flatten()
    predicted_masks_flat = predicted_masks_bin.flatten()

    # Calculate accuracy
    accuracy = accuracy_score(true_masks_flat, predicted_masks_flat)
    
    # Calculate precision (User's accuracy)
    precision = precision_score(true_masks_flat, predicted_masks_flat, average='binary')
    
    # Calculate recall (Producer's accuracy)
    recall = recall_score(true_masks_flat, predicted_masks_flat, average='binary')
    
    # Calculate F1-Score
    f1 = f1_score(true_masks_flat, predicted_masks_flat, average='binary')
    
    # Calculate Intersection over Union (IoU)
    iou_metric = MeanIoU(num_classes=2)
    iou_metric.update_state(true_masks, predicted_masks_bin)
    iou = iou_metric.result().numpy()
    
    # Calculate Dice Coefficient
    dice_coefficient = (2 * np.sum(true_masks_flat * predicted_masks_flat)) / (np.sum(true_masks_flat) + np.sum(predicted_masks_flat))
    
    # Store metrics in a dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "dice_coefficient": dice_coefficient
    }
    
    return metrics


def apply_confusion_matrix(true_masks, predicted_masks, threshold=0.5):
    """
    Apply a confusion matrix to U-Net predicted masks and true masks.
    
    Parameters:
    - true_masks (np.ndarray): Ground truth masks (binary).
    - predicted_masks (np.ndarray): Predicted masks (probability output from U-Net).
    - threshold (float): Threshold for binarizing predicted masks.
    
    Returns:
    - conf_matrix (np.ndarray): The confusion matrix.
    """
    
    # Step 1: Binarize predicted masks (if they are probabilities)
    predicted_masks_bin = (predicted_masks > threshold).astype(np.uint8)
    
    # Step 2: Flatten both masks
    true_masks_flat = true_masks.flatten()
    predicted_masks_flat = predicted_masks_bin.flatten()
    
    # Step 3: Compute the confusion matrix
    conf_matrix = confusion_matrix(true_masks_flat, predicted_masks_flat)
    
    return conf_matrix

def plot_confusion_matrix(cm, save_path):
    """
    Plot the confusion matrix.
    
    Parameters:
    - cm (np.ndarray): Confusion matrix to be plotted.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.show()


