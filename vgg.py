
import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import SSD

num_classes = 7

def create_model(num_classes):

    model=torchvision.models.detection.ssd300_vgg16(preTrained=True)

    for param in model.parameters():
        param.requires_grad = True

    num_default_boxes = len(model.anchor_generator.aspect_ratios[0]) * len(model.anchor_generator.scales)



    model.classification_headers = nn.ModuleList([
        nn.Conv2d(256, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(1024, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, num_classes * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, num_classes * num_default_boxes, kernel_size=3, padding=1)
    ])

    model.regression_headers = nn.ModuleList([
        nn.Conv2d(256, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(1024, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(512, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, 4 * num_default_boxes, kernel_size=3, padding=1),
        nn.Conv2d(256, 4 * num_default_boxes, kernel_size=3, padding=1)
    ])

    # Add custom layers on top of the pre-trained model
    custom_layers = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

    # Replace the final classification layer with custom layers
    model.classifier = custom_layers

    model.to('cpu')
    return model

def get_vgg_model():
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load('weights\model_vgg.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    
    return model

