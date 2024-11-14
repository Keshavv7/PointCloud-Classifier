import os
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import open3d as o3

from open3d.web_visualizer import draw # for non Colab

import matplotlib as mpl
import matplotlib.pyplot as plt

# TEMP for supressing pytorch user warnings
import warnings
warnings.filterwarnings("ignore")

# General parameters
NUM_TRAIN_POINTS = 2500
NUM_TEST_POINTS = 10000
NUM_CLASSES = 16
ROOT = r'C:\Users\itber\Documents\datasets\shapenetcore_partanno_segmentation_benchmark_v0'

# model hyperparameters
GLOBAL_FEATS = 1024

BATCH_SIZE = 32

# get class - label mappings
CATEGORIES = {
    'Airplane': 0, 
    'Bag': 1, 
    'Cap': 2, 
    'Car': 3,
    'Chair': 4, 
    'Earphone': 5, 
    'Guitar': 6, 
    'Knife': 7, 
    'Lamp': 8, 
    'Laptop': 9,
    'Motorbike': 10, 
    'Mug': 11, 
    'Pistol': 12, 
    'Rocket': 13, 
    'Skateboard': 14, 
    'Table': 15}
            
# Simple point cloud coloring mapping for part segmentation
def read_pointnet_colors(seg_labels):
    map_label_to_rgb = {
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [255, 0, 255],  # purple
        5: [0, 255, 255],  # cyan
        6: [255, 255, 0],  # yellow
    }
    colors = np.array([map_label_to_rgb[label] for label in seg_labels])
    return colors

from torch.utils.data import DataLoader
from shapenet_dataset import ShapenetDataset

# train Dataset & DataLoader
train_dataset = ShapenetDataset(ROOT, npoints=NUM_TRAIN_POINTS, split='train', classification=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation Dataset & DataLoader
valid_dataset = ShapenetDataset(ROOT, npoints=NUM_TRAIN_POINTS, split='valid', classification=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# test Dataset & DataLoader 
test_dataset = ShapenetDataset(ROOT, npoints=NUM_TEST_POINTS, split='test', classification=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# test Dataset  (segmentation version for display)
test_sample_dataset = ShapenetDataset(ROOT, npoints=NUM_TEST_POINTS, split='test', 
                                      classification=False, normalize=False)


sample_dataset = ShapenetDataset(ROOT, npoints=20000, split='train', 
                                 classification=False, normalize=False)


points, seg = sample_dataset[4000]

pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(points)
pcd.colors = o3.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))

o3.visualization.draw_plotly([pcd])

total_train_targets = []
for (_, targets) in train_dataloader:
    total_train_targets += targets.reshape(-1).numpy().tolist()

total_train_targets = np.array(total_train_targets)

class_bins = np.bincount(total_train_targets)

plt.bar(list(CATEGORIES.keys()), class_bins, 
             color=mpl.cm.tab20(np.arange(0, NUM_CLASSES)),
             edgecolor='black')
plt.xticks(list(CATEGORIES.keys()), list(CATEGORIES.keys()), size=12, rotation=90)
plt.ylabel('Counts', size=12)
plt.title('Train Class Frequencies', size=14, pad=20)

from pointnet import PointNetClassHead

points, targets = next(iter(train_dataloader))

classifier = PointNetClassHead(k=NUM_CLASSES, num_global_feats=GLOBAL_FEATS)
out, _, _ = classifier(points.transpose(2, 1))
print(f'Class output shape: {out.shape}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.optim as optim
from loss import PointNetLoss

EPOCHS = 100
LR = 0.0001
REG_WEIGHT = 0.001 

# use inverse class weighting
# alpha = 1 / class_bins
# alpha = (alpha/alpha.max())

# manually downweight the high frequency classes
alpha = np.ones(NUM_CLASSES)
alpha[0] = 0.5  # airplane
alpha[4] = 0.5  # chair
alpha[-1] = 0.5 # table

gamma = 2

optimizer = optim.Adam(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, 
                                              step_size_up=2000, cycle_momentum=False)
criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=REG_WEIGHT).to(DEVICE)

classifier = classifier.to(DEVICE)

mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)


def train_test(classifier, dataloader, num_batch, epoch, split='train'):
    ''' Function to train or test the model '''
    _loss = []
    _accuracy = []
    _mcc = []

    # return total targets and predictions for test case only
    total_test_targets = []
    total_test_preds = [] 
    for i, (points, targets) in enumerate(dataloader, 0):

        points = points.transpose(2, 1).to(DEVICE)
        targets = targets.squeeze().to(DEVICE)
        
        # zero gradients
        optimizer.zero_grad()
        
        # get predicted class logits
        preds, _, A = classifier(points)

        # get loss and perform backprop
        loss = criterion(preds, targets, A) 

        if split == 'train':
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # get class predictions
        pred_choice = torch.softmax(preds, dim=1).argmax(dim=1) 
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct.item()/float(BATCH_SIZE)
        mcc = mcc_metric(preds, targets)

        # update epoch loss and accuracy
        _loss.append(loss.item())
        _accuracy.append(accuracy)
        _mcc.append(mcc.item())

        # add to total targets/preds
        if split == 'test':
            total_test_targets += targets.reshape(-1).cpu().numpy().tolist()
            total_test_preds += pred_choice.reshape(-1).cpu().numpy().tolist()

        if i % 100 == 0:
            print(f'\t [{epoch}: {i}/{num_batch}] ' \
                  + f'{split} loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} mcc: {mcc:.4f}')
        
    epoch_loss = np.mean(_loss)
    epoch_accuracy = np.mean(_accuracy)
    epoch_mcc = np.mean(_mcc)

    print(f'Epoch: {epoch} - {split} Loss: {epoch_loss:.4f} ' \
          + f'- {split} Accuracy: {epoch_accuracy:.4f} ' \
          + f'- {split} MCC: {epoch_mcc:.4f}')

    if split == 'test':
        return epoch_loss, epoch_accuracy, epoch_mcc, total_test_targets, total_test_preds
    else: 
        return epoch_loss, epoch_accuracy, epoch_mcc
    


    # stuff for training
num_train_batch = int(np.ceil(len(train_dataset)/BATCH_SIZE))
num_valid_batch = int(np.ceil(len(valid_dataset)/BATCH_SIZE))

# store best validation mcc above 0.
best_mcc = 0.

# lists to store metrics (loss, accuracy, mcc)
train_metrics = []
valid_metrics = []

# TRAIN ON EPOCHS
for epoch in range(1, EPOCHS):

    ## train loop
    classifier = classifier.train()
    
    # train
    _train_metrics = train_test(classifier, train_dataloader, 
                                num_train_batch, epoch, 
                                split='train')
    train_metrics.append(_train_metrics)
        

    # pause to cool down
    time.sleep(4)

    ## validation loop
    with torch.no_grad():

        # place model in evaluation mode
        classifier = classifier.eval()

        # validate
        _valid_metrics = train_test(classifier, valid_dataloader, 
                                    num_valid_batch, epoch, 
                                    split='valid')
        valid_metrics.append(_valid_metrics)

        # pause to cool down
        time.sleep(4)

    # save model if necessary
    if valid_metrics[-1][-1] >= best_mcc:
        best_mcc = valid_metrics[-1][-1]
        torch.save(classifier.state_dict(), 'trained_models/cls_focal_clr_2/cls_model_%d.pth' % epoch)

metric_names = ['loss', 'accuracy', 'mcc']
_, ax = plt.subplots(len(metric_names), 1, figsize=(8, 6))

for i, m in enumerate(metric_names):
    ax[i].set_title(m)
    ax[i].plot(train_metrics[:, i], label='train')
    ax[i].plot(valid_metrics[:, i], label='valid')
    ax[i].legend()

plt.subplots_adjust(wspace=0., hspace=0.35)
plt.show()



MODEL_PATH = 'trained_models/cls_focal_clr/cls_model_35.pth'

classifier = PointNetClassHead(num_points=NUM_TEST_POINTS, num_global_feats=GLOBAL_FEATS, k=NUM_CLASSES).to(DEVICE)
classifier.load_state_dict(torch.load(MODEL_PATH))
classifier.eval()

num_test_batch = int(np.ceil(len(test_dataset)/BATCH_SIZE))

with torch.no_grad():
    epoch_loss, \
    epoch_accuracy, \
    epoch_mcc, \
    total_test_targets, \
    total_test_preds = train_test(classifier, test_dataloader, 
                              num_test_batch, epoch=1, 
                              split='test')
    
print(f'Test Loss: {epoch_loss:.4f} ' \
      f'- Test Accuracy: {epoch_accuracy:.4f} ' \
      f'- Test MCC: {epoch_mcc:.4f}')


from sklearn.metrics import confusion_matrix

test_confusion = pd.DataFrame(confusion_matrix(total_test_targets, total_test_preds),
                              columns=list(CATEGORIES.keys()),
                              index=list(CATEGORIES.keys()))





