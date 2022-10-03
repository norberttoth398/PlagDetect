from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000

#own functions
from .slicing import img_slice
from .tiling import tile_run
from .nms import mask_nms, matrix_nms
from .detector import detector
from .thresholding import threshold_labs, build_img

def __run__(img, path, config, checkpoint, device = "cpu", min_size_test = 1000, img_side = 2000, over_n = 100, nms_crit = 0.5):
    """Run entire script with.

    Args:
        weights (str): path to pretrained weights
        img (str): path to image to perform inference
        path (str): path to save everythin on; need to contain instance_res and panoptic_res subdirectories
        min_size_test (int, optional): Shortest side of individual tile to be resized to, recommend original divided by 2 if resolution is same as training set. Defaults to 1000.
        img_side (int, optional): Length of the side of individual tiles. Defaults to 2000.
        over_n (int, optional): Overlap between tiles. Defaults to 100.
    """
    
    model = detector(config, checkpoint, device)

    image  = cv2.imread(path + img)    

    n,m = img_slice(image, path, img_size=img_side, overlap=over_n)
    import glob
    import numpy as np
    images = glob.glob(path + '/imgs/*.jpg')

    for imageName in images:
        img_ = imageName
        name = imageName.split("/")[len(imageName.split("/"))-1]
        result = inference_detector(model, img_)
        if os.path.exists(path + "instance_res") == True:
            pass
        else:
            os.makedirs(path + "instance_res")

        #perform mask NMS for the first time
        labs = torch.ones((len(result[0][0])))
        scores = torch.Tensor(result[0][0][:,4])
        nms_res = mask_nms(torch.Tensor(np.asarray(result[1][0])), labs, scores)
        new_result = nms_remove(result, nms_res, nms_crit)

        #perform 2nd NMS to make sure everything is removed
        labs = torch.ones((len(new_result[0][0])))
        scores = torch.Tensor(new_result[0][0][:,4])
        nms_res = mask_nms(torch.Tensor(np.asarray(new_result[1][0])), labs, scores)
        new_result = nms_remove(new_result, nms_res, nms_crit)

        model.show_result(img_, result, out_file=path + "instance_res/" +name + ".jpg")

        label_img = tiling.create_label_image(new_result)
        
        if os.path.exists(path + "labels") == True:
            pass
        else:
            os.makedirs(path + "labels")
        np.savez(path + "labels/full_" + name, bb = new_result[0], mask = new_result[1])

    grid = (n,m)
    orig_shape = (image.shape[0], image.shape[1])
    img_side = img_side
    over_n = over_n
    tile_run_path = path + "labels"


    tiled_img, score_dict = tile_run(tile_run_path, grid, orig_shape, img_side, over_n)

    np.save(path + "labels/res.npy", tiled_img)
    np.savez(path + "labels/full_res.npy", img = tiled_img, scores = score_dict)

    #combine real img and panoptic res:
    img  = plt.imread(path + img)
    pan = np.load(path + "labels/res.npy")
    pan_mask = pan != 0
    pan_mask = pan_mask.astype("float")
    fig, ax = plt.subplots(1,1,figsize = (12,12))
    ax.imshow(img)
    ax.imshow(pan, alpha = pan_mask*0.5)
    fig.savefig(path + "labels/labelled_img", dpi = 500)


def __tile_only__(img_name, img_path,tile_path, out_path, img_side, overlap, thresh = 0):
    img = plt.imread(img_path + img_name + ".jpg")
    orig_shape = img.shape
    n, m = img_slice(img, img_path, img_size=img_side, overlap=overlap, slicing = False)
    grid = (n,m)

    tiled_img, score_dict = tile_run(tile_path, grid, orig_shape, img_side, overlap,score_thresh = thresh)

    np.savez(img_name + "_res.npy", img = tiled_img, scores = score_dict)
    fig, ax = plt.subplots(1,1,figsize = (12,12))
    ax.imshow(img)
    mask = tiled_img > 0
    ax.imshow(tiled_img, alpha = mask*0.5)
    fig.savefig(out_path + name +  "labelled_img_" + str(thresh), dpi = 500)
    return None
