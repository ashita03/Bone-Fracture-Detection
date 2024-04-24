import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSD
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

classes=['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
num_classes = 7

def get_model():
  
    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=7)
    model.load_state_dict(torch.load("weights\Resnet.pt", map_location='cpu'))
    
    return model


def get_vgg_model():
    
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False, num_classes=num_classes+1)
    model.load_state_dict(torch.load("weights\model_vgg.pt", map_location='cpu'))

    return model


def make_prediction(model, img, threshold):
    model.eval()
    #model.to(D).eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]
    
    return preds


def plot_image_from_output(img, annotation):
    
    img = img.cpu().detach().permute(1, 2, 0).numpy()    
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')
    
    class_name = None
    
    if annotation and "scores" in annotation and len(annotation["scores"]) > 0:
        max_score_idx = torch.argmax(annotation["scores"][0])

        # Extract the coordinates of the bounding box with the highest score
        xmin, ymin, xmax, ymax = annotation["boxes"][max_score_idx].detach().cpu().numpy()
        label_idx = annotation["labels"][max_score_idx]
        
        class_name = classes[label_idx]

        # Plot the bounding box with the highest score
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, class_name, fontsize=12, color='orange', fontweight='bold')

    return fig, ax, class_name

    
def figure_to_array(fig):

    fig.canvas.draw()
    
    return np.array(fig.canvas.renderer._renderer)