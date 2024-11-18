import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils2 import cxcywh_to_x1y1x2y2
from pathlib import Path

class FUNSDDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, annots_dir, width, height, transforms=None):
        self.transforms = transforms
        self.imgs_dir = Path(imgs_dir)
        self.annots_dir = Path(annots_dir)
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [
            image for image in sorted(os.listdir(imgs_dir)) if image[-4:] == ".png"
        ]

        # classes: 0 index is reserved for background
        self.classes = [None, "other", "question", "answer", "header"]

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = Path(self.imgs_dir) / img_name

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_filename = Path(img_name[:-4] + ".txt")
        annot_file_path = Path(self.annots_dir) / annot_filename
        print(f"{annot_file_path=}")
        boxes = []
        labels = []

        # box coordinates for xml files are extracted and corrected for image size given
        with open(annot_file_path, "r") as file:
            for line in file.readlines():
                class_id, cx, cy, w, h = list(map(float, line.strip().split(" ")))
                # Instead of image size, use the resized image size
                x1, y1, x2, y2 = cxcywh_to_x1y1x2y2((cx, cy, w, h), self.width, self.height)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:

            sample = self.transforms(
                image=img_res, bboxes=target["boxes"], labels=labels
            )

            img_res = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return img_res, target

    def __len__(self):
        return len(self.imgs)

root = Path("../datasets/funsd")
train_dir_imgs = root / "images" / "train"
train_dir_annots = root / "labels" / "train"
val_dir_imgs = root / "images" / "val"
val_dir_annots = root / "labels" / "val"
# check dataset
dataset = FUNSDDataset(train_dir_imgs, train_dir_annots, 640, 640)

# %%
# Function to visualize bounding boxes in the image
# Definitely not the problem - JC
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(10,10)
    a.imshow(img)
    for box in (target['boxes']):
        print(box)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()
    
def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

# %%
# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

dataset = FUNSDDataset(train_dir_imgs,train_dir_annots, 640, 640, transforms= get_transform(train=True))
dataset_val = FUNSDDataset(val_dir_imgs,val_dir_annots, 640, 640, transforms= get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_val = torch.utils.data.Subset(dataset_val, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_val, batch_size=4, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)

# %%
# to train on gpu if selected.
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

num_classes = 4 +1  # 4 classes + background

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.001,
                            weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

def main():
    '''Run if main module'''
    num_epochs = 10

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    # DONT MIND BELOW FOR NOW - JC
    # def apply_nms(orig_prediction, iou_thresh=0.3):
        
    #     # torchvision returns the indices of the bboxes to keep
    #     keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        
    #     final_prediction = orig_prediction
    #     final_prediction['boxes'] = final_prediction['boxes'][keep]
    #     final_prediction['scores'] = final_prediction['scores'][keep]
    #     final_prediction['labels'] = final_prediction['labels'][keep]
        
    #     return final_prediction

    # # function to convert a torchtensor back to PIL image
    # def torch_to_pil(img):
    #     return torchtrans.ToPILImage()(img).convert('RGB')

    # img, target = dataset_val[5]
    # # put the model in evaluation mode
    # model.eval()
    # with torch.no_grad():
    #     prediction = model([img.to(device)])[0]
        
    # print('predicted #boxes: ', len(prediction['labels']))
    # print('real #boxes: ', len(target['labels']))
    

if __name__ == '__main__':
    main()

# # %% [markdown]
# # Whoa! Thats a lot of bboxes. Lets plot them and check what did it predict

# # %%
# print('EXPECTED OUTPUT')
# plot_img_bbox(torch_to_pil(img), target)

# # %%
# print('MODEL OUTPUT')
# plot_img_bbox(torch_to_pil(img), prediction)

# # %% [markdown]
# # You can see that our model predicts a lot of bounding boxes for every apple. Lets apply nms to it and see the final output

# # %%
# nms_prediction = apply_nms(prediction, iou_thresh=0.2)
# print('NMS APPLIED MODEL OUTPUT')
# plot_img_bbox(torch_to_pil(img), nms_prediction)

# # %% [markdown]
# # Now lets take an image from the test set and try to predict on it

# # %%
# test_dataset = FUNSDDataset(test_dir, 480, 480, transforms= get_transform(train=True))
# # pick one image from the test set
# img, target = test_dataset[10]
# # put the model in evaluation mode
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])[0]
    
# print('EXPECTED OUTPUT\n')
# plot_img_bbox(torch_to_pil(img), target)
# print('MODEL OUTPUT\n')
# nms_prediction = apply_nms(prediction, iou_thresh=0.01)

# plot_img_bbox(torch_to_pil(img), nms_prediction)

# %% [markdown]
# The model does well on single object images.
# 
# You can see that our model predicts the slices too and that means a failure ☹️ . But fear not, this is just a base line model here are some ideas we can improve it - 
# 1. Use a better model. 
#    We have the option of changing the backbone of our model which at present is `resnet 50` and the fine tune it.
#    
# 2. We can change the training configurations like size of the images, optimizers and learning rate schedule.
# 3. We can add more augmentations.
#    We have used the Albumentations library which has an extensive library of data augmentation functions. Feel free to explore and try them out. 
