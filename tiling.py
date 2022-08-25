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

def create_label_image(result, score_thresh = 0):
    #create label image from result list (pre or post NMS)
    label_img = np.zeros((img_side,img_side))

    for i in range(len(result[1][0])):
        if result[0][0][i,4] < score_thresh:
            pass
        else:
            pred_mask = result[1][0][i]
            label_img_mask = label_img > 0
            #label_img_mask = label_img_mask.astype("int")

            new_lblImg = label_img_mask.astype("int") + pred_mask.astype("int")*2
            label_img[new_lblImg == 2] = i+1

    return label_img

def load_imgs(path, grid = (6,5), score_thresh = 0):
    #define relations between each tile for loading and post-processing

    n_tiles = grid[0]*grid[1]
    tile_list = []

    for n in range(n_tiles):
        data = np.load(path + "/full_image_" + str(int(n/grid[1])) + "_" + str(int(n%grid[1])) + ".jpg.npz")
        res = [data["bb"], data["mask"]]
        tile_list.append(create_label_image(res, score_thresh))

    return tile_list


def replace_vals(img1, img2, overlapped, corner_x, corner_y):
    coords = np.where(overlapped == 2)
    used_vals = []

    for i in range(len(coords[0])):
        coord = [coords[0][i], coords[1][i]]

        if img1[corner_y + coord[0], coord[1]+corner_x] == img2[coord[0], coord[1]]:
            new_val = img1[corner_y + coord[0], coord[1]+corner_x]
        else:
            if img2[coord[0], coord[1]] in used_vals:
                new_val = img2[coord[0], coord[1]]
                old_val = img1[coord[0] + corner_y, coord[1]+corner_x]
                img1[img1 == img1[coord[0] + corner_y, coord[1]+corner_x]] = new_val
                img2[img2 == old_val] = new_val
            else:
                new_val = img1[coord[0] + corner_y, coord[1]+corner_x]
                img2[img2 == img2[coord[0], coord[1]]] = new_val

        used_vals.append(new_val)

    return img1, img2


def left_overlap(i1, i2, corner_y, corner_x, over_n = 100, img_side = 2000):
    o1 = i1[corner_y:corner_y+img_side, corner_x:corner_x+over_n] #tile image
    o2 = i2[:,:over_n]

    mask1 = o1 != 0
    mask2 = o2 != 0
    overlapped = mask1.astype("int") + mask2.astype("int")

    i1, i2 = replace_vals(i1,i2,overlapped, corner_x, corner_y)

    overlapped[overlapped == 0] = 1
    i1[corner_y:corner_y + img_side, corner_x:corner_x+img_side] += i2
    i1[corner_y:corner_y + img_side, corner_x:corner_x+over_n] = i1[corner_y:corner_y + img_side, corner_x:corner_x+over_n]/overlapped

    return i1, i2

def top_overlap(i1, i2, corner_y, corner_x, over_n = 100, img_side = 2000, row_len = 9600):
    o1 = i1[corner_y:corner_y+over_n, corner_x:corner_x+img_side] #tile image
    o2 = i2[:over_n,:]

    mask1 = o1 != 0
    mask2 = o2 != 0
    overlapped = mask1.astype("int") + mask2.astype("int")

    i1, i2 = replace_vals(i1,i2,overlapped, corner_x, corner_y)

    overlapped[overlapped == 0] = 1
    i1[corner_y:corner_y + img_side, corner_x:corner_x+img_side] += i2
    i1[corner_y:corner_y + over_n, corner_x:corner_x+img_side] = i1[corner_y:corner_y + over_n, corner_x:corner_x+img_side]/overlapped

    return i1, i2

def top_overlap_row(i1, i2, corner_y, corner_x, over_n = 100, img_side = 2000, row_len = 9600):
    o1 = i1[corner_y:corner_y+over_n, corner_x:corner_x+row_len] #tile image
    o2 = i2[:over_n,:]

    mask1 = o1 != 0
    mask2 = o2 != 0
    overlapped = mask1.astype("int") + mask2.astype("int")
    #print(np.sum(o1[overlapped==2] - o2[overlapped==2]))

    i1, i2 = replace_vals(i1,i2,overlapped, corner_x, corner_y)

    #o1 = i1[corner_y:corner_y+over_n, corner_x:corner_x+row_len] #tile image
    #o2 = i2[:100,:]

    #mask1 = o1 != 0
    #mask2 = o2 != 0
    #new_overlapped = mask1.astype("int") + mask2.astype("int")
    #print(np.sum(o1[new_overlapped==2] - o2[new_overlapped==2]))
 
    overlapped[overlapped == 0] = 1
    i1[corner_y:corner_y + img_side, corner_x:corner_x+row_len] += i2
    i1[corner_y:corner_y + over_n, corner_x:corner_x+row_len] = i1[corner_y:corner_y + over_n, corner_x:corner_x+row_len]/overlapped

    return i1, i2

def tile_run(path,grid, orig_shape, img_side, over_n, score_thresh = 0):
    """Start the stitching process for the overall image from individual tiles post-inference.

    Args:
        path (_type_): _description_
        grid (_type_): _description_
        orig_shape (_type_): _description_
        img_side (_type_): _description_
        over_n (_type_): _description_

    Returns:
        _type_: _description_
    """
    tiled_img = np.zeros(orig_shape)
    row_len = orig_shape[1]
    tiled = load_imgs(path, grid, score_thresh)
    for j in range(grid[0]):
        
        tiled_row = np.zeros((img_side, orig_shape[1]))

        for i in range(grid[1]):
            
            current_tile_end = (i+1)*(img_side - over_n) + over_n
            if current_tile_end > orig_shape[1]:
                excess = current_tile_end - orig_shape[1]
            else:
                excess = 0

            #print(i)
            x = i#horizontal        

            #set top left corner at which they overlap
            corner_x = x*(img_side-over_n) - excess

            if i == 0:
                tiled_row[0:img_side, corner_x:corner_x + img_side] += tiled[j*grid[1]]
            else:
                
                #make sure no labels are the same across the two images
                new_img = tiled[(grid[1]*j)+i] + np.unique(tiled_row)[-1]
                new_img[new_img == np.unique(tiled_row)[-1]] = 0

                tiled_row, _ = left_overlap(tiled_row, new_img, 0, corner_x, over_n=over_n + excess ,img_side=img_side)
        
        
        current_row_end = (j+1)*(img_side - over_n) + over_n
        if current_row_end > orig_shape[0]:
            row_excess = current_row_end - orig_shape[0]
        else:
            row_excess = 0

        y = j#vertical
        corner_y = y*(img_side - over_n) - row_excess

        if j == 0:
            tiled_img[corner_y:corner_y+img_side, 0:row_len] += tiled_row
        else:
            tiled_row = tiled_row + np.unique(tiled_img)[-1]
            tiled_row[tiled_row == np.unique(tiled_img)[-1]] = 0
            tiled_img, _ = top_overlap_row(tiled_img, tiled_row, corner_y, 0, over_n=over_n+row_excess ,img_side = img_side, row_len=row_len)

    #optimizing label assignment
    n_unique = np.unique(tiled_img)
    for i in range(n_unique.size):
        tiled_img[tiled_img == n_unique[i]] = i  

    return tiled_img  