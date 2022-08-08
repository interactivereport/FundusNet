import sys
import os

class Configdir:
    def __init__(self):
        self.csv_dir = '/edgehpc/dept/compbio/users/whu1/fundusimage/data_kli3/csv_traintest'
        # self.output_dir = './results/round3-jun-26/'
        self.output_dir = '/edgehpc/dept/compbio/users/whu1/fundusimage/result_public'
        self.imgdirs = {'384': "/edgehpc/dept/compbio/users/whu1/fundusimage/data_kli3/imgdata/fundusimage_384_380",
                        '384_384': "/edgehpc/dept/compbio/users/whu1/fundusimage/data_kli3/imgdata/fundusimage_384_384",
                        '390_wx': '/edgehpc/dept/compbio/users/whu1/fundusimage/data_kli3/imgdata/fundusimage_390_wx',
                 '299': "/edgehpc/dept/compbio/users/whu1/fundusimage/data_kli3/imgdata/fundusimage_299",
                 '224': "/edgehpc/dept/compbio/users/whu1/fundusimage/data_kli3/imgdata/fundusimage_224",
                   '512': '/edgehpc/dept/compbio/projects/UKBB_fundus/preprocessed.fundus.512',
                   '1024': '/edgehpc/dept/compbio/projects/UKBB_fundus/preprocessed.fundus.1024',
                       'raw': '/edgehpc/dept/compbio/projects/UKBB_fundus/raw.fundus'}
        
    def get_img_dir(self, imgshape=0):
        if imgshape in self.imgdirs:
            return self.imgdirs[imgshape]
        else:
            sys.exit("Image Shape doesn't exist")
    
    def get_csv_dir(self):
        return self.csv_dir
    
    def get_output_dir(self):
        return self.output_dir