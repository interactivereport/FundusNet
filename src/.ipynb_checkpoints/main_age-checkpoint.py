import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from modelwrapper import Modelwrapper
from imgdataset import ImgDataset, ImgDataset_withaugment
from train_model import train_regression_model
from configdir import Configdir
cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args: --model --seed --imshape')
    parser.add_argument('-m','--model', help='model_name', type=str, required=True)
    parser.add_argument('-s','--seed', help='seed_num', type=str, required=True)
    parser.add_argument('-i','--imshape', help='image_shape', required=True)
    parser.add_argument('-b','--batch', help='batch_size', type=int, required=True)
    parser.add_argument('-e','--epoch', help='num_epoch', type=int, required=True)
    parser.add_argument('-p','--pheno', help='phenotype_name', type=str, required=True)
    parser.add_argument('--lrate', help='learning_rate', type=float, required=True)
    parser.add_argument('--loss_type', help='loss_type', type=str, required=True)
    parser.add_argument('--subsample', help='subsampling_train_data or not? 0/percentage', type=int, required=True)
    parser.add_argument('--r_degree', help='rotate_degree_train_data', type=int, required=True)
    parser.add_argument('-v','--version', help='version number', type=str, required=True)
    args = vars(parser.parse_args())

    dirobj = Configdir()
    img_dir = dirobj.get_img_dir(imgshape=args['imshape'])
    csv_dir = dirobj.get_csv_dir()
    output_dir = dirobj.get_output_dir()
    if args['subsample'] > 0 and args['subsample'] < 100:
        model_out_name = os.path.join(output_dir, "model_wts", args['pheno']+"_"+args['model']+"_seed_"+args['seed']+"_v_"+args['version']+'_subsample'+str(args['subsample'])+".pth")
    else:
        model_out_name = os.path.join(output_dir, "model_wts", args['pheno']+"_"+args['model']+"_seed_"+args['seed']+"_v_"+args['version']+".pth")
            
        
    
    if os.path.exists(model_out_name): sys.exit(f'{model_out_name}  exists, please check your arguments')

    csv_files = {x: os.path.join(csv_dir, args['pheno']+"_inlier_seed_"+args['seed']+"_"+x+".csv") for x in ['train', 'val', 'test']}
    image_datasets = {x: ImgDataset_withaugment(csv_file=csv_files[x], root_dir=img_dir, crop='center')
                  for x in ['val', 'test']}
    image_datasets['train'] = ImgDataset_withaugment(csv_file=csv_files['train'], root_dir=img_dir, resample=args['subsample'], crop='rand', rodegree=args['r_degree'])

    print(f"pheno={args['pheno']}, model={args['model']}, loss={args['loss_type']}, lr={args['lrate']}, seed={args['seed']}, batsize={args['batch']}, imgsize={args['imshape']}, \
          \n v={args['version']}, subsample_train={args['subsample']}, rotate_degree_train={args['r_degree']}, \n model.pth_out=={model_out_name}")

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args['batch'], shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mwrapper = Modelwrapper(num_classes=1)
    cmodel = getattr(mwrapper, args['model'])().to(device)

    criterion = nn.L1Loss() if args['loss_type']=='MAE' else nn.MSELoss()
    optimizer_ft = optim.Adam(cmodel.parameters(), lr=args['lrate'])

    # Decay LR by a factor of 0.5 every 5 epochs
    stepdivid = 7 if args['epoch'] > 14 else 5
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=round(args['epoch']/stepdivid), gamma=0.7)

    model_args = {"model": cmodel, "criterion": criterion, "optimizer": optimizer_ft, "scheduler": exp_lr_scheduler, "num_epochs": args['epoch'],
                  "dataloaders": dataloaders, "dataset_sizes": dataset_sizes, 'model_name': args['model']}
    [model_ft, best_val_loss] = train_regression_model(**model_args)
    torch.save(model_ft.state_dict(), model_out_name)