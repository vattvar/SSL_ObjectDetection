import torchvision
import torch
import os
import numpy as np
import torch
from PIL import Image
import yaml
from common_functions import * 
from torchvision import datasets, models, transforms
from torch.nn import CrossEntropyLoss,MSELoss
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn
#import transforms as T
from torchvision import datasets, models, transforms
from engine import *
from utils import *
import time
from torch.utils.data import Subset
import albumentations as A
from torch.utils.tensorboard import SummaryWriter

train_data_path = "/labeled/labeled/training/"
val_data_path = "/labeled/labeled/validation/"

writer = SummaryWriter()

def custom_collate_fn(batch):
    return tuple(zip(*batch))

class ObjectDetection(torch.utils.data.Dataset):
    def __init__(self, root,isTrain,transform1,transform2):
        self.root = root
        self.transform1 = transform1
        self.images_folder = "images"
        self.labels_folder = "labels"
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.images_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, self.labels_folder))))
        self.transform2  = transform2
        self.isTrain = isTrain

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.images_folder, self.imgs[idx])
        target_path = os.path.join(self.root, self.labels_folder, self.masks[idx])
        targets = None
        with open(target_path, 'r') as file:
            targets = yaml.safe_load(file)
        img = Image.open(img_path).convert("RGB")

        #---- TARGETS------
        target_return = {}
        target_return['boxes'] = torch.as_tensor(targets['bboxes'],dtype=torch.float32)
        target_return['labels'] = torch.as_tensor([label2id_encode(cat) for cat in targets['labels']])
        target_return['area'] = torch.as_tensor([(box[2]-box[0])*(box[3]-box[1]) for box in targets['bboxes']],dtype=torch.float32)
        target_return['iscrowd']  = torch.zeros((len(targets['bboxes']),), dtype=torch.int64)
        # print(self.masks[idx])
        # target_return['image_id'] = torch.as_tensor([int(self.masks[idx][:-4])],dtype=torch.int)
        target_return['image_id'] = torch.tensor([idx])


        if self.transform1 is not None:
            if(self.isTrain):
                op = self.transform2(image=np.array(img), bboxes=target_return['boxes'], class_labels=target_return['labels'])
                target_return['boxes'] = torch.as_tensor(op["bboxes"],dtype=torch.float32)
                target_return['labels'] = torch.as_tensor(op["class_labels"])
                img = op["image"]

            img = self.transform1(img)

        return img, target_return

    def __len__(self):
        return len(self.imgs)


"""
https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/blob/main/models/fasterrcnn_convnext_small.py
"""
def create_model(res50_custom,num_classes=100):
    # Load the pretrained features.

    backbone = torch.nn.Sequential(*(list(res50_custom.children())[:-1]))
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    backbone.out_channels = 2048

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model



transform2 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))

transform1 = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# We then pass the original dataset and the indices we are interested in
dataset = ObjectDetection(train_data_path,True,transform1,transform2)
# train_indices = list(range(15000,20000))
# dataset = Subset(dataset, train_indices)
dataset_test = ObjectDetection(val_data_path,False,transform1,None)
# test_indices = list(range(1000,1500))
# dataset_test = Subset(dataset_test, test_indices)

'''Step1 : RPN Training'''
resnet50 = torch.load('models/best_backbone_2022-12-05.pt')
model = create_model(resnet50,get_num_classes())

''' Step2: ROI Training '''
# model = torch.load("models/bestObjDet_RPN.pt")
# backbone = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
# backbone.out_channels = 2048
# model.backbone = backbone
# model = torch.load("bestObjDet_7.pt")



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,num_workers=2, collate_fn=custom_collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

checkpoint_bb = torch.load('models/pretext2.pt')
model_bb = models.resnet50()
model_bb.load_state_dict(checkpoint_bb['model_state_dict'])

# move model to the right device
checkpoint = torch.load('models/best_stage3.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.backbone = torch.nn.Sequential(*(list(model_bb.children())[:-1]))
checkpoint_epoch = checkpoint['epoch']
model = model.to(device)

# checkpoint = torch.load('models/best_ROI.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# model_rpn = torch.load('models/best_RPN.pt')
# model.rpn = model_rpn.rpn
# checkpoint_epoch = 0
# del model_rpn
# model.to(device)


# construct an optimizer
# params = [p for p in model.backbone.parameters()] + [p for p in model.rpn.parameters() ]
params = [p for p in model.backbone.parameters()] + [p for p in model.roi_heads.parameters() ]
# params =  [p for p in model.parameters()]




optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# let's train it for 10 epochs
num_epochs = 100
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

since = time.time()
for epoch in tqdm(range(checkpoint_epoch,num_epochs)):

    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch,100,writer)
    time_elapsed = time.time() - since
    print(time_elapsed)
    # update the learning rate
    lr_scheduler.step()
    print('Done')
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
    time_elapsed = time.time() - since
    print(time_elapsed)
    # torch.save(model,f"bestObjDet_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'models/best_new.pt')

print("That's it!")






# totalTrainLoss = 0.0
    # trainCorrect = 0.0

    # for (images, targets) in tqdm(data_loader):
    #     images = list(image.to(device) for image in images)
    #     print(images[0].shape)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		#     # perform a forward pass and calculate the training loss
    #     predictions = model(images,targets)
    #     print(predictions)
    #     bboxLoss = bboxLossFunc(predictions[0], targets['boxes'])
    #     classLoss = classLossFunc(predictions[1], targets['labels'])
    #     totalLoss = (bboxLoss) + (classLoss)
		# # zero out the gradients, perform the backpropagation step,
		# # and update the weights
    #     optimizer.zero_grad()
    #     totalLoss.backward()
    #     optimizer.step()
    #     lr_scheduler.step()
		# # add the loss to the total training loss so far and
		# # calculate the number of correct predictions
    #     totalTrainLoss += totalLoss
    #     trainCorrect += (predictions[1].argmax(1) == targets['labels']).type(torch.float).sum().item()

    # print(totalTrainLoss/len(dataset))
    # print(trainCorrect/len(dataset))
