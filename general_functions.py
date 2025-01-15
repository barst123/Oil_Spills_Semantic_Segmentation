import matplotlib.pyplot as plt
import numpy as np
import json, requests, os
import glob
import cv2
import gc

def scale_array(array):
    """

    Description:

    This function scales image to grayscale (0-255)

    Parameters:

    array - image array to scale to grayscale (0-255)
    """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = 255 * (array - min_val) / (max_val - min_val)
    return scaled_array

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

def get_path_to_save(file_path):
    """
    
    Description:

    This function gets path from input file path to save the image

    Parameters:

    file_path - path to save the file
    """
    img_paths=file_path.split('/')
    name_paths=img_paths[10].split('-')
    name=img_paths[5]+'_'+img_paths[6]+'_'+img_paths[7]+'_'+name_paths[0]+'_'+name_paths[1]+'_'+name_paths[2]+'_'+name_paths[3]+'_'+name_paths[6]+name_paths[7]+'.jpg'
    path='/home/eouser/Desktop/oil_spills/images/sar/'+name
    del img_paths, name_paths, name
    return path

def create_image_for_labeling(path_list, crop_patch=1024, kernel_size=25, n=0, k=0, sap_filtr=13):
    """
    
    Description:

    This function saves the processed image in grayscale

    Parameters:

    path_list - path list form url

    crop_patch - width and height of patch images (1024)

    kernel_size - size of kernel to equalize histogram (25)

    n - Y-coordinates of the upper left corner * crop_patch

    k - X-coordinates of the upper left corner * crop_patch

    sap_filtr - size of kernel to fitr salt and pepper noise (13)
    """

    for dir_path in path_list:
        matching_dir=glob.glob(os.path.join(dir_path, 'measurement'))
        for dir in matching_dir:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith('001.tiff'): #only first tiff file
                        file_path=glob.glob(os.path.join(dir, file))[0] #importing path
                        
                        '''operations on the original image'''
                        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        x_start, y_start=k*crop_patch, n*crop_patch #cooridnates of upper left corner
                        x_end, y_end=(image.shape[1]//crop_patch)*crop_patch, (image.shape[0]//crop_patch)*crop_patch #coordinates of bottom right corner
                        image=image[y_start:y_end, x_start:x_end] #image cropping
                        del x_start, y_start, x_end, y_end
                        path_save=get_path_to_save(file_path)

                        '''image normalization to grayscale (0-255) and operation on it'''
                        scaled=scale_array(image) #image scaling
                        scaled = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #conversion from uint16 to uint8
                        del image

                        '''increase photo contrast'''
                        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(kernel_size, kernel_size)) #histogram equalization
                        clahe_scaled = clahe.apply(scaled)
                        del scaled

                        '''denoising an image'''
                        median_image=cv2.medianBlur(clahe_scaled, sap_filtr) #salt and pepper noise filtering
                        del clahe_scaled
                        print(path_save)
                        plt.imshow(median_image, cmap='gray')
                        plt.show()
                        
                        cv2.imwrite(path_save, median_image)
                        del median_image

def get_images(start_date, end_date, point_x, point_y):
    """
    
    Description:

    This function gets images that intersect point within a defined time interval
    
    Parameters:

    start_date - date in format YYYY-MM-DD

    end_date - date in format YYYY-MM-DD

    point_x - X-coordinates of POI in WGS84

    point_y - Y-coordinates of POI in WGS84
    """
    images=[]
    crop_patch=1024
    k=0
    n=0
    kernel_size=25
    sap_filtr=13
    url=f'https://datahub.creodias.eu/odata/v1/Products?$filter=((ContentDate/Start%20ge%20{start_date}T00:00:00.000Z%20and%20ContentDate/Start%20le%20{end_date}T23:59:59.999Z)%20and%20(Online%20eq%20true)%20and%20(OData.CSC.Intersects(Footprint=geography%27SRID=4326;POINT%20({point_x}%20{point_y})%27))%20and%20(((((Collection/Name%20eq%20%27SENTINEL-1%27)%20and%20(((Attributes/OData.CSC.StringAttribute/any(i0:i0/Name%20eq%20%27productType%27%20and%20i0/Value%20eq%20%27GRD%27)))))))))&$expand=Attributes&$expand=Assets&$orderby=ContentDate/Start%20asc&$top=20'

    products=json.loads(requests.get(url).text)
    path_list=[]
    for item in products['value']:
        path_list.append(item['S3Path'])
    del url
    del products

    for dir_path in path_list:
        matching_dir=glob.glob(os.path.join(dir_path, 'measurement'))
        for dir in matching_dir:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    file_path=None
                    if file.endswith('001.tiff'): #only first tiff file
                        file_path=glob.glob(os.path.join(dir, file))[0] #importing path
                    if file_path:
                        '''operations on the original image'''
                        image=cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        x_start, y_start=k*crop_patch, n*crop_patch #cooridnates of upper left corner
                        x_end, y_end=(image.shape[1]//crop_patch)*crop_patch, (image.shape[0]//crop_patch)*crop_patch #coordinates of bottom right corner
                        image=image[y_start:y_end, x_start:x_end] #image cropping
                        del x_start, y_start, x_end, y_end

                        '''image normalization to grayscale (0-255) and operation on it'''
                        scaled=scale_array(image) #image scaling
                        scaled = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #conversion from uint16 to uint8
                        del image

                        '''increase photo contrast'''
                        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(kernel_size, kernel_size)) #histogram equalization
                        clahe_scaled = clahe.apply(scaled)
                        del scaled

                        '''denoising an image'''
                        median_image=cv2.medianBlur(clahe_scaled, sap_filtr) #salt and pepper noise filtering
                        del clahe_scaled

                        images.append(median_image)
    return images

def combine_patches(image, patches, patch_size):
    """
    
    Description:

    This function combines small images into an image the size of the original image

    Parameters:

    image - original image

    patches - patch images array

    patch_size - the size of the width and height of a single patch image
    """
    height=image.shape[0]
    width=image.shape[1]
    image=np.zeros((height, width), dtype=patches[0].dtype)
    patch_idx = 0
    
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            image[i:i + patch_size, j:j + patch_size] = patches[patch_idx]
            patch_idx += 1
            
    return image

def predictions(images, model, crop_patch, batch_size):
    """"
    
    Description:

    This function predicts masks for input images

    Parameters:

    images - array of input images

    model - U-net model

    crop_patch - width and height of a patch images

    batch_size - number of batch in a model
    """
    predicted_images=[]
    for image in images:
        image_patches=make_patches(image, crop_patch) #making patches from image
        image_patches=np.array(image_patches)
        image_patches=np.expand_dims(image_patches, axis=-1) #adding another dimension
        img_predict=model.predict(image_patches, batch_size=batch_size) #predicting mask for each image
        img_predict=np.squeeze(img_predict, axis=-1) #removal of added dimension
        reconstructed=combine_patches(image, img_predict, crop_patch) #combinig patches into an image
        predicted_images.append(reconstructed)
        del image_patches, img_predict, reconstructed
        gc.collect()
    return predicted_images