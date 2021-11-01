SIMPLE_NN_V2 EAMPLES

This doc explain how run simple_nn_v2 example code for SiO2

More Informations in 
https://simple-nn.readthedocs.io/en/latest/examples/examples.html

For each direcory from 1 to 4 execute run.sh bash file to run


# 0. default input files 

### OUTCAR $\rightarrow$ Calculated output files 

### structure_list $\rightarrow$ file that contain directory of OUTCARs to create symmetry function  

### input.yaml $\rightarrow$ all settings for run SIMPLE_NN_v2 in yaml

### params_Si, params_O $\rightarrow$ sysmmetry function hyperparameters




# 1. generate_data
- In this example create symmetry function of configurations as binary file using torch.save function  

input.yaml in 1.generate_data folder
```yaml
generate_features: true  # generate data with descriptor
preprocess: false
train_model: false
params: # set of parameters that used in generating process
    Si: params_Si 
    O:  params_O
descriptor:
    types: 'symmetry_function' # discriptor types that will be generated
    absolute_path: False #This desable writing output file as relative path 
```
## Return 

### ./data/{numbers}.pt $\rightarrow$  generated symmetry functions of single configuration

### total_list $\rightarrow$ file that contains directory of generated files

# 2.preprocess
- In this example split train, valid set from generated dataset and calculate scale factor & pca & gdf(if needed)

## Input
### total_list $\rightarrow$ file that contains directory of generated files, need to split train, valid set calculating scale factor, pca, gdf  

```yaml
generate_features: false
preprocess: true #Set true to preprocess generated files
train_model: false
params:
    Si: params_Si
    O:  params_O

preprocessing: 
    valid_rate: 0.1 #Valid ratio from total_list, other ratio are setted to train set 
     calc_scale: True #Calculate scale facrtor
     calc_pca: True #Calculate Pinciple Component Analysis for symmetry functions
```

### Return
### train_list, valid_list  $\rightarrow$ file that contains directory of train sets and valid sets  
### scale_factor $\rightarrow$ binary file that contain value that used in scaling symmetry functions
### pca $\rightarrow$ binary file that contain informations PCA vectors and eigenvalues


# 3.train_model
- In this example train the neural network model using generated data & preprocessed data
### Input
### ./data/*.pt $\rightarrow$ generated symmetry functions to use train model  
### train_list, valid_list $\rightarrow$ directory of train, valid sets
### pca & scale_factor $\rightarrow$ need if use scaling, PCA during training 
```yaml
generate_features: false
preprocess: false
train_model: true #Set to train model
params:
    Si: params_Si
    O:  params_O

neural_network:
     optimizer: #Set what optimizer to use (implemented in pytorch)
         method: 'Adam'
     nodes: '30-30' #Specify Network Nodes the total nodes are (# of SFs)*(node informations) 
     batch_size: 10 #Specify batch size in dataloader
     total_iteration: 1000 #Total epoch to train models
     learning_rate: 0.001 #Total epoch to train models
     pca: True
     scale: True


```



1. check error
In 3.check_error folder

5.parameter tuning for GDF (Optional)
In 4.parameter_tuning_GDF


