import torch
import torch.nn as nn

cfg = {
    'vgg5': [64, 'M', 128, 'M', 256, 'M'],
    'vgg6': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'vgg7': [64, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'vgg8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']
}

numchan = {'vgg5': 256, 'vgg6': 512, 'vgg7':512, 'vgg8': 512}

class myVGG(nn.Module):
    def __init__(self, vgg_name, num_classes=2):
        super(myVGG, self).__init__()
        self.num_classes = num_classes
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=numchan[vgg_name]*5*5, out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)