import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

from imgdataset import ImgDataset_withaugment

def my_show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5,
                      cam_cutoff: float =0.2) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1]")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    cam = np.where(np.stack([mask, mask, mask], axis=-1)<cam_cutoff, img, cam)
    img_tmp = img.sum(axis=-1)
    img_tmp = np.stack([img_tmp, img_tmp, img_tmp], axis=-1)*255
    cam = np.where(img_tmp<50, img, cam)
    return np.uint8(255 * cam)

def input_img(image):
    input_image = cv2.normalize(image.squeeze().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input_image = input_image.swapaxes(0, 1).swapaxes(1, 2).astype(np.int32)
    return input_image

def raw_img(imgname, img_dir):
    raw_image = np.asarray(Image.open(os.path.join(img_dir, imgname)))
    raw_image = raw_image[3:-3, 3:-3, :]
    return raw_image
    
def plot_figures(pheno_name: str, figures: list, lbls: list, imgnames: list, pheno: list, result_dir: str, i_row: int=0):
    nrows, ncols = len(figures), len(figures[0])
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    cfig = plt.gcf()
    cfig.set_size_inches(ncols*5, nrows*5) # 10,14 for ms
    for irow in range(nrows):
        for icol in range(ncols):
            axeslist[irow, icol].imshow(figures[irow][icol]) #, cmap=plt.gray()
            if irow==0:
                figurename = imgnames[icol]+'\n'+pheno+'='+str(lbls[icol].item())
                axeslist[irow, icol].set_title(figurename, fontsize=25)
            axeslist[irow, icol].set_axis_off()
            plt.tight_layout() # optional
    print(imgnames)
    # 
    plt.savefig(os.path.join(result_dir, f'{pheno_name}_row_{str(i_row+1)}_gcam.png'), dpi=60)
    return

def run_save_gcam_results(cam,
                      img_dir,
                      result_dir,
                      csv_file,
                      pheno_name):

    batchsize = 1
    cam_cutoff = 0.5 if pheno_name=='glaucoma' else 0.2

    imgs, labels, imgnames = [], [], []

    image_datasets = ImgDataset_withaugment(csv_file=csv_file, root_dir=img_dir, crop='center')
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize, num_workers=4)
    dataloader_iterator = iter(dataloader)

    # *rawimgs* => image_datasets transform => *imgs* => transform back => *imgs_input*    
    n_cols = min(4, len(dataloader))
    n_rows = math.ceil(len(dataloader)/n_cols)

    for i in range(len(dataloader)): 
        img, label, imgname = next(dataloader_iterator)
        imgs.append(img)
        labels.append(label)
        imgnames.append(imgname[0])

    for i_row in range(n_rows):
        imgsinput, imgsraw, gcams, camonimgs= [], [], [], []
        i_start = i_row*n_cols
        i_end = i_start + n_cols

        for i in range(i_start, i_end):
            imgsraw.append(raw_img(imgnames[i], img_dir))
            imgsinput.append(input_img(imgs[i]))
            gcams.append(cam(input_tensor=imgs[i]).squeeze())
            camonimgs.append(my_show_cam_on_image(img=imgsinput[i-i_start]/255, mask=gcams[i-i_start], use_rgb=True, cam_cutoff=cam_cutoff))

        plot_figures(pheno_name=pheno_name,
                     figures=[imgsraw, camonimgs],
                     lbls=labels[i_start:i_end],
                     imgnames=imgnames[i_start:i_end],
                     pheno=pheno_name,
                     result_dir=result_dir,
                     i_row=i_row)
    return