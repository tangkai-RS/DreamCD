import os
import torch
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class ChangeAnywhereBase(Dataset):
    def __init__(
        self,
        data_csv,
        size=None,
        random_crop=False,
        interpolation="bicubic",
        n_labels=8,
        downsample_size=64,
        only_building=False,
        only_generate=False,
        with_adain=True,
    ):
        
        self.n_labels = n_labels
        self.data_csv = data_csv
        self.only_building = only_building
        self.only_generate = only_generate
        self.with_adain = with_adain
        
        self.img_A_path = []
        self.img_B_path = []
        self.label_A_path = []
        self.label_B_path = []
        self.change_mask_path = []
        self.img_B_syn_path = [] # the path list of generated images
        with open(self.data_csv, "r") as f:
            for line in f:
                paths = line.strip().split(" ")
                self.img_A_path.append(paths[0])
                self.img_B_path.append(paths[1])
                self.label_A_path.append(paths[2])
                self.label_B_path.append(paths[3])
                self.change_mask_path.append(paths[4])
                self.img_B_syn_path.append(paths[5])
                
        self._length = len(self.img_A_path)
        
        self.samples = {
            "img_A_path": self.img_A_path,
            "img_B_path": self.img_B_path,
            "label_A_path": self.label_A_path,
            "label_B_path": self.label_B_path,
            "change_mask_path": self.change_mask_path,
            "img_B_syn_path": self.img_B_syn_path
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.change_mask_rescaler = albumentations.SmallestMaxSize(max_size=downsample_size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        sample = dict((k, self.samples[k][i]) for k in self.samples)
        
        img_A = None if self.only_generate else np.asarray(Image.open(sample["img_A_path"])).astype(np.uint8) 
        img_B = None if not self.with_adain else np.asarray(Image.open(sample["img_B_path"])).astype(np.uint8)
        label_A = None if self.only_generate else np.asarray(Image.open(sample["label_A_path"])).astype(np.uint8)
        label_B = np.asarray(Image.open(sample["label_B_path"])).astype(np.uint8)
        # label_B[label_B==2] = 4
 
        if self.only_generate:
            change_mask = None
        elif self.only_building:
            change_mask = (label_A == label_B).astype(np.uint8)
            change_mask_orig = (label_A == label_B).astype(np.uint8)
            # change_mask = np.zeros_like(change_mask).astype(np.uint8)
            # change_mask_orig = np.zeros_like(change_mask).astype(np.uint8)
            # TODO: mod!!!!
            # change_mask_orig = np.asarray(Image.open(sample["change_mask_path"])).astype(np.uint8) # single channel for change mask
            # change_mask_orig = np.where(change_mask_orig==255, 0, 1)
        else:
            change_mask = np.asarray(Image.open(sample["change_mask_path"])).astype(np.uint8)             
            change_mask = np.where(change_mask==255, 0, 1) # change == 0, unchanged == 1  
            # change_mask = np.zeros_like(change_mask).astype(np.uint8)
            change_mask_orig = change_mask.copy()
            
        if self.size is not None:
            label_B = self.segmentation_rescaler(image=label_B)["image"]
            if (img_A is not None) and (label_A is not None) and (img_B is not None):
                img_A = self.image_rescaler(image=img_A)["image"]
                img_B = self.image_rescaler(image=img_B)["image"]
                label_A = self.segmentation_rescaler(image=label_A)["image"]
                processed = self.preprocessor(
                    image=img_A,
                    image_B=img_B,
                    label_A=label_A,
                    label_B=label_B,
                )   
                sample["img_A"] = (processed["image"]/127.5 - 1.0).astype(np.float32)  
                sample["img_B"] = (processed["image_B"]/127.5 - 1.0).astype(np.float32)  
                label_A = processed["label_A"]
                onehot_A = np.eye(self.n_labels)[label_A]
                sample["label_A"] = onehot_A
                
                label_B = processed["label_B"] 
                label_B_ds = self.change_mask_rescaler(image=label_B)["image"]
                onehot_B = np.eye(self.n_labels)[label_B]        
                sample["segmentation"] = onehot_B # condition mask      
                sample["label_B_ds"] = np.expand_dims(label_B_ds, 0)  
                label_A_ds = self.change_mask_rescaler(image=label_A)["image"]
                sample["label_A_ds"] = np.expand_dims(label_A_ds, 0)  
                
            elif (img_A is not None) and (label_A is not None):
                img_A = self.image_rescaler(image=img_A)["image"]
                label_A = self.segmentation_rescaler(image=label_A)["image"]
                processed = self.preprocessor(
                    image=img_A,
                    label_A=label_A,
                    label_B=label_B,
                )   
                sample["img_A"] = (processed["image"]/127.5 - 1.0).astype(np.float32)  
                label_A = processed["label_A"]
                onehot_A = np.eye(self.n_labels)[label_A]
                sample["label_A"] = onehot_A
                
                label_B = processed["label_B"] 
                label_B_ds = self.change_mask_rescaler(image=label_B)["image"]
                onehot_B = np.eye(self.n_labels)[label_B]        
                sample["segmentation"] = onehot_B # condition mask      
                sample["label_B_ds"] = np.expand_dims(label_B_ds, 0)     
                label_A_ds = self.change_mask_rescaler(image=label_A)["image"]
                sample["label_A_ds"] = np.expand_dims(label_A_ds, 0)   
                          
            else:
                processed = self.preprocessor(image=label_B)    
                label_B = processed["image"] 
                onehot_B = np.eye(self.n_labels)[label_B]        
                sample["segmentation"] = onehot_B 
        else:
            processed = {
                        "img_A": img_A,
                        "img_B": img_B,
                        "label_A": label_A,
                        "label_B": label_B,
                         }
              
        if change_mask is not None:
            sample["change_mask_orig"] = np.expand_dims(change_mask_orig, 0) # keep orig spatial resolution
            change_mask = self.change_mask_rescaler(image=change_mask)["image"]    
            sample["change_mask"] = np.expand_dims(change_mask, 0)
        return sample
    

class ChangeAnywhere2(ChangeAnywhereBase):
    def __init__(self, data_csv, size=256, random_crop=False, interpolation="bicubic", only_building=False,
                 only_generate=False, with_adain=True):
        super().__init__(data_csv=data_csv,
                         size=size, 
                         random_crop=random_crop, 
                         interpolation=interpolation,
                         only_building=only_building,
                         only_generate=only_generate,
                         with_adain=with_adain)
        

def class2RGB(class_map):
    CLASS_RGB_VALUES = [
        [128, 0, 0],
        [0, 255, 36],
        [148, 148, 148],
        [255, 255, 255],
        [34, 97, 38],
        [0, 69, 255],
        [75, 181, 73],
        [222, 31, 7]
    ] 
    class_map = class_map.squeeze()
    shape = class_map.shape
    if len(shape) > 2:
        b, h, w = shape
        img_rgbs = []
        for j in range(b):
            class_map_temp = class_map[j, :, :]
            img_rgb = np.zeros([h, w, 3]).astype(np.uint8)    
            for i in range(len(CLASS_RGB_VALUES)):
                img_rgb[class_map_temp==i, :] = CLASS_RGB_VALUES[i]            
            img_rgbs.append(img_rgb.transpose(2, 0, 1))
        return np.stack(img_rgbs)
    else:
        h, w = shape
        img_rgb = np.zeros([h, w, 3]).astype(np.uint8)
        for i in range(len(CLASS_RGB_VALUES)):
            img_rgb[class_map==i, :] = CLASS_RGB_VALUES[i]
        return img_rgb