SIMPLE_NN_V2 EXAMPLES

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
## Output
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

## Output
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
     total_epoch: 1000 #Total epoch to train models
     learning_rate: 0.001 #Total epoch to train models
     pca: True
     scale: True
```
## Output
### checkpoint_bestmodel.pth.tar $\rightarrow$ binary file that contains best model information during traning  
### checkpoint_latest.pth.tar $\rightarrow$ binary file that contains latest model
### potential_saved_bestmodel $\rightarrow$ lammps potential file that contains best model parameters
### potential_saved_latest $\rightarrow$ lammps potential file that contains latest model parameters

# 4.error_check
- In this example check geneated neural network model error by using generated data
## Input 
### checkpoint.tar $\rightarrow$ Binary file that contains model parameter  loaded in test  
###  test_list $\rightarrow$ text file that contains generated data directory  list  
```yaml
generate_features: false
preprocess: false
train_model: true

params:
    Si: params_Si
    O:  params_O

neural_network:
    nodes: '30-30'
    batch_size: 10
    train: false #Should be false to test model
    test: true #Should be true to test model
    continue: 'checkpoint.tar' #File to load during test
```
## Output
### test_reuslt $\rightarrow$ saved test result values,you can read with torch.load function

#  5.parameter_tuning_GDF 
In 5.parameter_tuning_GDF you can use force weighted model depends on density of Symmetry fucntions. 

For detailed information

[W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790][https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.8b08063]

Using GDF in training is two step.

## 1. Calculate GDF value and save to generated files
## No additional input needed, but need additional information to yaml
```yaml
#input_gdf.yaml
generate_features: false
preprocess: true
train_model: false
params:
    Si: params_Si
    O:  params_O

preprocessing:
    valid_rate: 0.1
     calc_scale: True   #Must be true due to GDF uses scale factor 
     calc_pca: True     #Must be true due to GDF uses PCA
     calc_gdf: True     #Set true to calculate GDF values 
     atomic_weights:    #This gives information about GDF
       type: gdf        #Newly calculate GDF values
       params:
         sigma: Auto    #Automatic calculation of sigma values
```
## Output 
### atomic_weights $\rightarrow$ binary file that contains information of calculated atomic weights, It used with weight modifier  
### The generated GDF value for each Symmetry functions are saved to generated *.pt files with key 'gdf'

## 2. Traning with GDF value & weight modifier
## Input
### *.pt $\rightarrow$ Generated symmetry functions with GDF values
### atomic_weights $\rightarrow$ binary file that contains GDF value informations

```yaml
generate_features: false
preprocess: false
train_model: true
params:
  Si: params_Si
  O: params_O

neural_network:
  optimizer:
    method: Adam
  nodes: 30-30
  batch_size: 10
  total_epoch: 100
  learning_rate: 0.001
  use_force: true
  pca: true
  scale: true
  gdf: true #Set true to use GDF in traning 
  weight_modifier: #Set weight modifier that convert GDF value
    type: modified sigmoid #Set weight modifier type
    params: #Set weight modifier parmeters for each species
      Si:
        b: 1.
        c: 35.
      O:
        b: 1.
        c: 74.
```
### During traning, weight modifier calculate modified GDF value, and the value is  weight of force in loss function

### If you set structure weight for configuration, the value is ignored and GDF used instead
