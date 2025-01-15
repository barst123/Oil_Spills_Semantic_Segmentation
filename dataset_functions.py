import numpy as np
import dask.array as da
import os
import cv2
import random
import gc
import psutil

from dask import delayed, compute
from pathlib import Path

crop_patch=1024
random.seed(123)
image_dataset=[]
mask_dataset=[]
imgs_dir='/home/eouser/Desktop/oil_spills/images/sar'
masks_dir='/home/eouser/Desktop/oil_spills/images/masks/png'
oil_dataset=[]
non_oil_dataset=[]
oil_dataset_mask=[]
non_oil_dataset_mask=[]


@delayed
def make_patches(image, patch_size):
    """"
    
    Description:

    This function creates patches from input image

    Parameters:

    image - input image

    patch_size - width and height of patch images
    """
    height, width = image.shape[:2]
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches

@delayed
def image_process(image, mask, crop_patch):
    """
    
    Description:

    This function processes image and mask (rotates and flips) and dividing them into smaller patches. It adds images and masks with at least one pixel value>0 to the "with oil" datasets, and remaining images and masks to the "without oil" datasets

    Parameters:

    image - input image

    mask - input mask

    crop_patch - width and height of patch images
    """
    oil_dataset=[]
    oil_dataset_mask=[]
    non_oil_dataset=[]
    non_oil_dataset_mask=[]
    transformations=[
        (da.from_array(image, chunks=(crop_patch, crop_patch)), da.from_array(mask, chunks=(crop_patch, crop_patch))), #original image and mask
        (da.from_array(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), chunks=(crop_patch, crop_patch)), da.from_array(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE), chunks=(crop_patch, crop_patch))), #rotate the image and mask by 90 deg
        (da.from_array(cv2.rotate(image, cv2.ROTATE_180), chunks=(crop_patch, crop_patch)), da.from_array(cv2.rotate(mask, cv2.ROTATE_180), chunks=(crop_patch, crop_patch))), #rotate the image and mask by 180 deg
        (da.from_array(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), chunks=(crop_patch, crop_patch)), da.from_array(cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE), chunks=(crop_patch, crop_patch))), #rotate the image and mask by 270 deg
        (da.from_array(cv2.flip(image, 1), chunks=(crop_patch, crop_patch)), da.from_array(cv2.flip(mask, 1), chunks=(crop_patch, crop_patch))), #horizontal mirroring of image and mask
        (da.from_array(cv2.flip(image, 0), chunks=(crop_patch, crop_patch)), da.from_array(cv2.flip(mask, 0), chunks=(crop_patch, crop_patch))) #vertical mirroring of image and mask
    ]
    image_patches=[]
    mask_patches=[]
    for image_transformed, mask_transformed in transformations:
        image_patches.extend(compute(make_patches(image_transformed, crop_patch))[0])
        mask_patches.extend(compute(make_patches(mask_transformed, crop_patch))[0])

    for image_patch, mask_patch in zip(image_patches, mask_patches):
        if np.max(mask_patch)>0:
            oil_dataset.append(image_patch)
            oil_dataset_mask.append(mask_patch)
        else:
            non_oil_dataset.append(image_patch)
            non_oil_dataset_mask.append(mask_patch)

    del transformations

    return oil_dataset, oil_dataset_mask, non_oil_dataset, non_oil_dataset_mask

@delayed
def read_image(file_path):
    """
    
    Description:

    This function loads the input image

    Parameters:

    file_path - path to input image
    """
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

@delayed
def read_mask(file_path):
    """
    
    Description:

    This function loads the input mask and converts it to binary mask
    
    Parameters:

    file_path - path to input mask
    """
    mask=cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    mask=np.where(mask>0, 1, mask)
    return mask

def dask_dataset(image_dataset=image_dataset, mask_dataset=mask_dataset):
    """
    
    Description

    This function creates dataset from images and masks by dividing them into smaller patches. It checks how many patches have at least one pixel with value>0 and randomly selects the same number of patches in which no pixel has value>0.

    Parameters:

    image_dataset - empty array to store the image dataset

    mask_dataset - empty array to store the mask dataset
    """
    # Process images and masks
    for subdir, dirs, files in os.walk(imgs_dir):
        if 'done' in subdir:
            files.sort()
            for file in files:
                try:
                    img_path = os.path.join(subdir, file)
                    mask_path = None

                    for mask_subdir, mask_dirs, mask_files in os.walk(masks_dir):
                        if 'png' in mask_subdir:
                            for mask_file in mask_files:
                                if mask_file == Path(file).stem + '_mask.png':
                                    mask_path = os.path.join(mask_subdir, mask_file)
                                    break
                        if mask_path:
                            break

                    if mask_path:
                        print(f"Processing file: {file}")
                        image = read_image(img_path)
                        maska = read_mask(mask_path)

                        print("Running image_process...")
                        data = compute(image_process(image, maska, crop_patch))[0]
                        oil, mask_oil, non_oil, mask_non_oil = data

                        del image, maska
                        gc.collect()

                        if len(non_oil) > 0 and len(oil) > 0:
                            rand_ind = random.sample(range(len(non_oil)), len(oil))
                            non_oil = [non_oil[i] for i in rand_ind]
                            mask_non_oil = [mask_non_oil[i] for i in rand_ind]

                        print(len(oil), len(non_oil))
                            
                        image_dataset.append(oil)
                        mask_dataset.append(mask_oil)
                        image_dataset.append(non_oil)
                        mask_dataset.append(mask_non_oil)

                        del oil, non_oil, mask_oil, mask_non_oil, data
                        gc.collect()

                        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
                        
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue

    print("Finished processing all images.")

    try:
        print("Combining datasets...")
        results = compute(*image_dataset, *mask_dataset)
        #print(results)
        image_results = results[:len(image_dataset)]
        mask_results = results[len(image_dataset):]
        print(len(results), len(image_results), len(mask_results))

        image_dataset = da.concatenate([da.from_array(np.array(patches), chunks=(len(patches), crop_patch, crop_patch)) for patches in image_results])
        mask_dataset = da.concatenate([da.from_array(np.array(patches), chunks=(len(patches), crop_patch, crop_patch)) for patches in mask_results])
        del image_results
        del mask_results
        image_dataset=image_dataset.compute()
        mask_dataset=mask_dataset.compute()

        print("Datasets combined successfully.")
        print(f"Image dataset shape: {image_dataset.shape}", len(image_dataset))
        print(f"Mask dataset shape: {mask_dataset.shape}", len(mask_dataset))

    except Exception as e:
        print(f"Error combining datasets: {e}")
    
    return image_dataset, mask_dataset