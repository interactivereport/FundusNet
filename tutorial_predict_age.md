```python
import os
import numpy as np
import pandas as pd

import torch

from modelwrapper import Modelwrapper
from imgdataset import ImgDataset_withaugment
```


```python
# specify input files
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'regnety_32'
pheno_name = 'age'# 'gender'

cmodel_ckpt = f'../model_ckpt/{pheno_name}_{model_name}_v0.pth'
csv_file = f'../result_biomarker_interpretation/input_imgs/imgs_{pheno_name}.csv'
img_dir = '../result_biomarker_interpretation/input_imgs/'
result_dir = '../result_biomarker_interpretation/'

batchsize = 2
```


```python
num_classes = 1 if pheno_name=='age' else 2
mwrapper = Modelwrapper(num_classes=num_classes)
cmodel = getattr(mwrapper, model_name)().to(device)
cmodel.load_state_dict(torch.load(cmodel_ckpt))
cmodel.eval()

image_datasets = ImgDataset_withaugment(csv_file=csv_file, root_dir=img_dir, crop='center')
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize, num_workers=4)
dataloader_iterator = iter(dataloader)

```


```python
img_names = []
y_true = []
y_preds = []

for inputs, labels, img_name in dataloader:
    img_names.extend(list(img_name))
    y_true.extend(labels.tolist())
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = cmodel(inputs)
    if num_classes==2:
        _, preds = torch.max(preds, 1)
    y_preds.extend(preds.cpu().detach().numpy().flatten())

df = pd.DataFrame({'filename':img_names, 'class':y_true, f'predicted_{pheno_name}':y_preds})
df.to_csv(os.path.join(result_dir, f'imgs_{pheno_name}_predicted.csv'), index=False)
print(df.head())
```

                 filename  class  predicted_age
    0  image_0_age_61.png     61      58.069225
    1  image_1_age_61.png     61      62.588211
    2  image_2_age_72.png     72      73.754265
    3  image_3_age_53.png     53      53.196304

