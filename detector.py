from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000

class detector:
    def __init__(self,file = 'models/mmdet_detectoRS_dataAug.py', checkpoint = 'models/27May_r50_detectoRS_extraAug.pth', device = "cpu"):

        # Specify the path to model config and checkpoint file
        config_file = file
        checkpoint_file = checkpoint

        # build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device=torch.device(device))
    

    def forward():
        return model