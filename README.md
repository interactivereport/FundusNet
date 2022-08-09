# FundusNet
To to list:
> Run different CNN models on fundus image data (seed 101, 102, 103, 104 generated using cross-validation)

### step 1
Change seed number (shage.sh -> line 14 -> --seed)

> Jake:  --seed 101

> Jeron: --seed 102

> Ye:    --seed 103

> Talia: --seed 104

### step 2
Change model name (shage.sh -> line 14 -> --model). 

Each time, pick one model from
```
dm_nfnet_f2  eca_nfnet_l2  regnety_32  xcit_m_384  volo_d2_384  regnet_x_32gf
```

### step 3
Run command line
```
sbatch shage.sh
```

In the end, your results (best_model_weights) will be saved in ``` /edgehpc/dept/compbio/users/whu1/fundusimage/result_public/model_wts ``` 
For example, Jake will generate model_weights for 
> data --seed 101
> > --model dm_nfnet_f2  
> > --model eca_nfnet_l2  
> > --model regnety_32  
> > --model xcit_m_384  
> > --model volo_d2_384  
> > --model regnet_x_32gf
