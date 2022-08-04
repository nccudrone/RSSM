import albumentations as albu
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
import glob
import os
from torchvision import transforms
import time

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation_real():
    test_transform = [
        albu.Resize(320, 320)
    ]
    return albu.Compose(test_transform)

def get_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = get_validation_augmentation_real()(image=img)['image']
    img = get_preprocessing(preprocessing_fn)(image=img)['image']

    x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    return pr_mask

# model config
CLASSES = ['road']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ENCODER = 'se_resnext101_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# overlay param
alpha = 0.5

# load model 
model = torch.load('model/road_model.pth')
model.eval()

test_dir = 'test/'
# read images from test
allFileNames = os.listdir(test_dir)

# show image for debug
show_image_flag = True
show_mask_flag = True

start_time = time.time()
with torch.no_grad():
    for name in allFileNames:
        print(name)
        img = cv2.imread(test_dir + name)
              
        if show_image_flag == True:
            img_viz = img.copy()

        pr_mask = get_prediction(img)

        if show_mask_flag == True:
            cv2.imshow("mask", pr_mask)

        if show_image_flag == True:

            overlay = np.zeros((320, 320, 3), dtype = np.float32)
            overlay[:,:,0] = pr_mask

            img_viz = np.float32(img_viz)/255.0
            img_viz = cv2.resize(img_viz, (overlay.shape[0], overlay.shape[1]), interpolation=cv2.INTER_CUBIC)

            cv2.addWeighted(overlay, alpha, img_viz, 1, 0, img_viz)
            cv2.imshow("overlay", img_viz)
            #print(img_path)
            cv2.imwrite(test_dir+name.split(".")[0] + "_predict.png", img_viz*255)
                
        if show_mask_flag == True or show_image_flag == True:
            cv2.waitKey(10)

print("All prediction time:", time.time() - start_time)
