import torch
import numpy as np
import pickle
import cv2
from skimage.metrics import structural_similarity as ssim


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def mySSIM(tar_img, prd_img):
    """Calculate SSIM for PyTorch tensors using scikit-image"""
    # Convert tensors to numpy arrays
    tar_img_np = tar_img.detach().cpu().numpy().transpose(1, 2, 0)
    prd_img_np = prd_img.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Ensure the values are in [0, 1]
    tar_img_np = np.clip(tar_img_np, 0, 1)
    prd_img_np = np.clip(prd_img_np, 0, 1)
    
    # Calculate SSIM
    return ssim(tar_img_np, prd_img_np, channel_axis=2, data_range=1.0)

def batch_SSIM(img1, img2, average=True):
    """Calculate SSIM for a batch of images"""
    SSIM = []
    for im1, im2 in zip(img1, img2):
        ssim = mySSIM(im1, im2)
        SSIM.append(ssim)
    return sum(SSIM)/len(SSIM) if average else sum(SSIM)