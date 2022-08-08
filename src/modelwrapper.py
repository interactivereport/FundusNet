import torch
import torch.nn as nn
from torchvision import models
from myvgg import myVGG
import timm
# https://github.com/d-li14/efficientnetv2.pytorch

class Modelwrapper:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.p_drout = 0.5
    
    ## resnets and densenets don't have dropout, but it is fine to add it
    def resnest269e(self):
        model = timm.create_model('resnest269e', pretrained=True, num_classes=self.num_classes)
        return model
    
    def deit_base_dist_384(self):
        model = timm.create_model('deit_base_distilled_patch16_384', pretrained=True, num_classes=self.num_classes)
        return model
    
    def deit_base_384(self):
        model = timm.create_model('deit_base_patch16_384', pretrained=True, num_classes=self.num_classes)
        return model
    
    def vit_base_384(self):
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=self.num_classes)
        return model
    
    def vit_base_r50_384(self):
        model = timm.create_model('vit_base_r50_s16_384', pretrained=True, num_classes=self.num_classes)
        return model
    
    def cait_s24_384(self):
        model = timm.create_model('cait_s24_384', pretrained=True, num_classes=self.num_classes)
        return model
    
    def beit_base_384(self):
        model = timm.create_model('beit_base_patch16_384', pretrained=True, num_classes=self.num_classes)
        return model
    
    def eca_nfnet_l2(self):
        model = timm.create_model('eca_nfnet_l2', pretrained=True, num_classes=self.num_classes)
        return model
    
    def dm_nfnet_f2(self):
        model = timm.create_model('dm_nfnet_f2', pretrained=True, num_classes=self.num_classes)
        return model
    
    def dm_nfnet_f3(self):
        model = timm.create_model('dm_nfnet_f3', pretrained=True, num_classes=self.num_classes)
        return model
    
    def volo_d2_384(self):
        model = timm.create_model('volo_d2_384', pretrained=True, num_classes=self.num_classes)
        return model

    def volo_d3_448(self):
        model = timm.create_model('volo_d3_448', pretrained=True, num_classes=self.num_classes)
        return model
    
    def xcit_m_384(self):
        model = timm.create_model('xcit_medium_24_p16_384_dist', pretrained=True, num_classes=self.num_classes)
        return model
    
    def regnetz_4h(self):
        model = timm.create_model('regnetz_040h', pretrained=True, num_classes=self.num_classes)
        return model
    
    # def regnetx_32(self):   # should be equivalent to regnet_x_32gf
    #     model = timm.create_model('regnetx_320', pretrained=True, num_classes=self.num_classes)
    #     return model
    
    def regnety_32(self):
        model = timm.create_model('regnety_320', pretrained=True, num_classes=self.num_classes)
        return model
    
    def nf_regnet_b5(self):
        model = timm.create_model('nf_regnet_b5', pretrained=True, num_classes=self.num_classes)
        return model
    
    def regnetz_d32(self):
        model = timm.create_model('regnetz_d32', pretrained=True, num_classes=self.num_classes)
        return model
       
    def eff_v2_s(self):
        model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=self.num_classes)
        return model
    
    def eff_v2_m(self):
        model = timm.create_model('efficientnetv2_rw_m', pretrained=True, num_classes=self.num_classes)
        return model
    
    def vgg5(self):
        model = myVGG(vgg_name='vgg5', num_classes=self.num_classes)
        return model
    
    def vgg6(self):
        model = myVGG(vgg_name='vgg6', num_classes=self.num_classes)
        return model
    
    def vgg7(self):
        model = myVGG(vgg_name='vgg7', num_classes=self.num_classes)
        return model
    
    def alexnet(self):
        model = models.alexnet(pretrained=True)
        layers = [
            nn.Dropout(p=model.classifier[0].p, inplace=True),
            nn.Linear(in_features=model.classifier[1].in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
        
    def resnet18(self):
        model = models.resnet18(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
    
    def resnet50(self):
        model = models.resnet50(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
        
    def resnet101(self):
        model = models.resnet101(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
    
    def resnet152(self):
        model = models.resnet152(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
        
    def resnext50_32x4d(self):
        model = models.resnext50_32x4d(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
    
    def resnext101_32x8d(self):
        model = models.resnext101_32x8d(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
        
    def densenet121(self):
        model = models.densenet121(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.classifier.in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def densenet169(self):
        model = models.densenet169(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.classifier.in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def densenet201(self):
        model = models.densenet201(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.classifier.in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def inception_v3(self):
        model = models.inception_v3(pretrained=True) # inception has a dropout layer before fc
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        return model
    
    def efficientnet_b4(self):
        model = models.efficientnet_b4(pretrained=True)
        layers = [
            nn.Dropout(p=model.classifier[0].p, inplace=True),
            nn.Linear(in_features=model.classifier[1].in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def efficientnet_b5(self):
        model = models.efficientnet_b5(pretrained=True)
        layers = [
            nn.Dropout(p=model.classifier[0].p, inplace=True),
            nn.Linear(in_features=model.classifier[1].in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def vgg11_bn(self):
        model = models.vgg11_bn(pretrained=True)
        layers = [
            nn.Dropout(p=model.classifier[2].p, inplace=True),
            nn.Linear(in_features=model.classifier[0].in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def vgg16_bn(self):
        model = models.vgg16_bn(pretrained=True)
        layers = [
            nn.Dropout(p=model.classifier[2].p, inplace=True),
            nn.Linear(in_features=model.classifier[0].in_features, out_features=self.num_classes, bias=True)
        ]
        model.classifier = nn.Sequential(*layers)
        return model
    
    def regnet_x_8gf(self):
        model = models.regnet_x_8gf(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
    
    def regnet_x_32gf(self):
        model = models.regnet_x_32gf(pretrained=True)
        layers = [
            nn.Dropout(p=self.p_drout),
            nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes, bias=True)
        ]
        model.fc = nn.Sequential(*layers)
        return model
    