#!/bin/python
#
#This code for validate symmetry function as pytorch save file
#Need to do 
#Convert atomic position to symmetry function 
#Check sf value in torch & pickle and validate
#
import os, sys
import subprocess as sbp
import ase
from ase import io
from ase import units
from ase import neighborlist as nb
import torch
import pickle
import six
import yaml
import numpy as np
from numpy import pi


##Set precision using numpy ##
np.set_printoptions(precision = 8)

##### File processing #####
#load pickle file(tempoaray) at util.__init__
def load_pickle(filename):
    with open(filename, 'rb') as f:
        if six.PY2:
            return pickle.load(f)
        elif six.PY3:
            return pickle.load(f, encoding='latin1')

                                                 
#read parameters from file at symmetry_function.__init__
# Parameter formats
# [type of SF(1)] [atom type index(2)] [cutoff distance(1)] [coefficients for SF(3)]
def read_params(filename):
    params_i = list()
    params_d = list()
    with open(filename, 'r') as fil:
        for line in fil:
            tmp = line.split()
            params_i += [list(map(int,   tmp[:3]))]
            params_d += [list(map(float, tmp[3:]))]                
        params_i = np.asarray(params_i, dtype=np.intc, order='C')
        params_d = np.asarray(params_d, dtype=np.float64, order='C')  
    return params_i, params_d

#Get parameters file directory
def get_params_dir(filename):
    params_dir = list()
    input_yaml = read_yaml(filename)
    atom_type = input_yaml['atom_types']
    for atom in atom_type:
        params_dir.append(input_yaml['symmetry_function']['params'][atom])
    return atom_type , params_dir

#Read VASP-OUTCAR file (only applicable VASP-OUTCAR format) 
#At Symmetry_function.generate in symmetry_function.__init__
def open_outcar(filename):
    if ase.__version__ >= '3.18.0':
        snapshot = io.read(filename, index=-1, format='vasp-out')
    else:
        snapshot = io.read(filename, index=-1, format='vasp-out', force_consistent=True)
    return snapshot

#Read yaml file 
def read_yaml(filename):
    with open(filename) as f:
        output = yaml.safe_load(f)
    return output

##### Symmetry function processing #####
#Run symmetry functions as input ase.atoms
def _cutoff_func(dist, cutoff):
    out = np.cos(pi*dist/cutoff) + 1.0
    out = 0.5*np.float64(out)
    out = np.float64(out)
    return out

#Gaussian function(G2) to use
def _gaussian(dist, cutoff, eta, r_s = 0):
    out = np.exp(-eta*(dist-r_s)**2)*_cutoff_func(dist , cutoff)
    out = np.float64(out)
    return out

#Decorator function / preprocess distance & angle of atomic environemnt needed 
#index -> SF type & param_d -> parameters of SF
def generate_sf(index, param_d):
    cutoff = param_d[0]
    if index == 2: 
        eta = param_d[1]; r_s = param_d[2]
        # G2 SF 
        def sym_func(dist_list):
            output = 0
            for dist in dist_list:
                output += _gaussian(dist,cutoff,eta)
            output = np.float64(output)
            return output
    elif index == 4: 
        eta = param_d[1]; zeta = param_d[2] ; lamda = param_d[3]
        # G4 SF 
        def sym_func(dist_list):
            output = 0
            angle = 0
            for dist in dist_list:    # dist = [dist_ij,dist_ik ,dist_jk]
                if dist[2] < cutoff:  #Cufoff for distance Rjk
                    #G4 symmetry function
                    tmp = 0
                    dist_ij , dist_ik , dist_jk = dist
                    angle_rad = np.arccos((dist_ij**2+dist_ik**2-dist_jk**2)/(2*dist_ij*dist_ik))
                    tmp  = _gaussian(dist_ij, cutoff , eta)*_gaussian(dist_ik, cutoff , eta)*_gaussian(dist_jk, cutoff , eta)
                    tmp *= (1 + lamda * np.cos(angle_rad))**zeta
                    tmp  = np.float64(tmp)
                    output += tmp
            output *= 2.0**(1-zeta)
            output = np.float64(output)  
            return output
    elif index == 5: 
        eta = param_d[1]; zeta = param_d[2] ; lamda = param_d[2]
        # G5 SF 
        def sym_func(dist_list):
            output = 0
            for dist in dist_list:   # dist = [dist_ij,dist_ik, dist_jk] type
                tmp = 0
                #G5 symmetry function 
                dist_ij , dist_ik , dist_jk = dist                                            
                angle_rad = np.arccos((dist_ij**2+dist_ik**2-dist_jk**2)/(2*dist_ij*dist_ik))
                tmp  = _gaussian(dist_ij, cutoff , eta)*_gaussian(dist_ik, cutoff , eta)
                tmp *= (1 + lamda * np.cos(angle_rad))**zeta
                tmp  = np.float64(tmp)
                output += tmp
            output *= 2.0**(1-zeta)
            output = np.float64(output) 
            return output
    return sym_func

### OUTCAR Processing ###
#Class that contains information of atoms in OUTCAR
class Distance_atoms:
    def __init__(self, atoms):
        self.atoms = atoms
        self.atom_sym = np.asarray(atoms.get_chemical_symbols())
        self.atom_dist= self.atoms.get_all_distances(mic = True)           
        atoms.pbc = True ##  Use PBC condition ##
        ## Need to do --> PBC over three cell problematic : distance + lattice vector/2 ? 
        self.atom_num_dict = dict()
        self.atom_num_tot = len(self.atom_sym)
        for atom in self.atom_sym:
            try:
                self.atom_num_dict[atom] += 1
            except:
                self.atom_num_dict[atom] = 1
        self.atom_name = list(self.atom_num_dict.keys())
        #Culumative number of atom to use distance check of species
        tmp = 0
        self.atom_cul_num = list()
        self.atom_cul_dict = dict()
        for name in self.atom_name:
            self.atom_cul_dict[name] = tmp
            tmp += self.atom_num_dict[name]
            self.atom_cul_num.append(tmp)
        self.cutoff = 0  #Default setting initiaizaing with zero

    def check_params(self):
        print('Atom Sylbom',self.atom_sym)
        print('Atom Distance matrix',self.atom_dist)
        print('Atom index',self.atom_name)
        print('Atom number dictionary',self.atom_num_dict)


    def set_cutoff(self, cutoff):
        if not (np.abs(self.cutoff- cutoff) < 1E-6):  # Chek cufoff distance 
            self.first_index, self.second_index, self.distance = nb.neighbor_list(\
            quantities = 'ijd', cutoff = cutoff,  a = self.atoms, self_interaction = False)
        self.cutoff = cutoff

    def get_g2_distance(self, num, index_i):
       i_name = self.atom_name[index_i-1] # Get species name 
       num_atom_bool = self.first_index == (num-1) #Get bool array to make distance list
       num_atom_list = self.second_index[num_atom_bool]
       num_distance = self.distance[num_atom_bool]
       bool_start = (self.atom_cul_dict[i_name] <= num_atom_list)
       bool_end   = (self.atom_cul_dict[i_name] + self.atom_num_dict[i_name] > num_atom_list)
       index_bool = np.logical_and(bool_start , bool_end) # Bool list of specific species
       dist_out = num_distance[index_bool]
       return dist_out

    def _get_tri_dist(self, permut):
        if permut != None:
            i , j  = permut
            out = [self._num_distance[self._num_atom_list == i],\
                   self._num_distance[self._num_atom_list == j],\
                   self.atom_dist[i,j]]
            out = np.float64(out)
        else:
            out = [0,0,0]
        return out

    def _get_atom_name(self, number):
        for name in self.atom_name:
            if self.atom_cul_dict[name] <= (number-1) and self.atom_cul_dict[name] + self.atom_num_dict[name] > (number-1):
                return name


    def get_g4_distance(self, num, index_i, index_j):
        if index_i == index_j:
            i_name = self.atom_name[index_i-1]
            same = True
        else:
            i_name = self.atom_name[index_i-1]
            j_name = self.atom_name[index_j-1]
            same = False
        num_atom_bool = (self.first_index == (num-1)) #Temporary 
        self._num_atom_list = self.second_index[num_atom_bool]
        self._num_distance  = self.distance[num_atom_bool]
        #Define dictionary to use permutation distances 
        permut_dict = dict()
        for atom in self.atom_name: #initialization
            permut_dict[atom] = list()
        for i in self._num_atom_list:
            permut_dict[self._get_atom_name(i+1)].append(i)
        #Create permutation invariant list
        permut_list = list()
        if same:
            atom_list = permut_dict[i_name]
            length = len(permut_dict[i_name])
            if length > 1: #length must be larger than 1
                for i in range(length):
                    for j in range(i+1,length):
                        permut_list.append([atom_list[i],atom_list[j]])
            else:
                permut_list.append(None)
        else:  ## different index of i & j
            if any(permut_dict[i_name]) and any(permut_dict[j_name]):
                for atom_i in permut_dict[i_name]:
                    for atom_j in permut_dict[j_name]:
                        permut_list.append([atom_i,atom_j])
            else:
                permut_list.append(None)
        #Get distance of three atom
        dist_out = list()
        for permut in permut_list:
            dist_out.append(self._get_tri_dist(permut))
        self._num_atom_bool = None
        self._num_atom_list = None
        self._num_distance  = None
        return  dist_out


#Class that generate list of symmetry function 
class Test_symmetry_function:
    def __init__(self, output_name = None, yaml_name = None, data_name = None):
        ## Open file and make Distance_atoms class
        if output_name != None:
            self.structure = open_outcar(output_name) 
            self.distance  = Distance_atoms(self.structure) # Load Distance_atoms structure
            self.atom_type =  self.distance.atom_name
            self.atom_number = self.distance.atom_num_dict
        if yaml_name != None:
            self.atom_type , params_dir = get_params_dir(yaml_name)
            self.params_dict = dict()
            self.symmetry_function_dict = dict()
            self.length = dict()
            for num , atom in enumerate(self.atom_type):
                par_i , par_d = read_params(params_dir[num])
                self.params_dict[atom] = [par_i , par_d]
                #Generate Symmetry function list from parameters
                sf_list = list()
                tmp = None # Temporary function
                for i ,  params in enumerate(par_d):
                    tmp = generate_sf(index = int(par_i[i][0]) , param_d  = params)
                    sf_list.append(tmp)
                self.length[atom] = len(sf_list)
                self.symmetry_function_dict[atom] =  sf_list
        if data_name != None:
            if data_name.split('.')[-1] == 'pt':
                self.data = torch.load(data_name)
                self.data_sf = self.data['x']
            elif data_name.split('.')[-1] == 'pickle':
                self.data =  load_pickle(data_name)
                self.data_sf = self.data['x']
    def set_structure(self, output_name):
        self.structure = open_outcar(output_name) 
        self.distance  = Distance_atoms(self.structure) # Load Distance_atoms structure
        self.atom_type =  self.distance.atom_name
        self.atom_number = self.distance.atom_num_dict

    def set_yaml(self, yaml_name):
        self.atom_type , params_dir = get_params_dir(yaml_name)
        self.params_dict = dict()
        self.symmetry_function_dic = dict()
        for num , atom in enumerate(self.atom_type):
            par_i , par_d = read_params(params_dir[num])
            self.params_dict[atom] = [par_i , par_d]
            sf_list = list()
            for i ,  params in enumerate(par_d):
                sf_list.append(generate_sf(index = par_i[i][0] , param_d  = par_d))
            self.symmetry_function_dict[atom] =  sf_list

    def set_data(self, data_name): 
        if data_name.split('.')[-1] == 'pt':
            self.data = torch.load(data_name)
            self.data_sf = self.data['x']
        elif data_name.split('.')[-1] == 'pickle':
            self.data =  load_pickle(data_name)
            self.data_sf = self.data['x']

    #Setting parameters by hands Need output_name & param_dir
    def set_params(self, atom, params_dir):
        #Declariation
        self.params_dict = dict()
        self.symmetry_function_dic = dict()
        sf_list = list()
        par_i , par_d = read_params(params_dir[num])
        self.params_dict[atom] = [par_i , par_d]
        for i ,  params in enumerate(par_d):
            sf_list.append(generate_sf(index = par_i[i][0] , param_d  = par_d))
        self.symmetry_function_dict[atom] =  sf_list
 
    def show_data_info(self):
        print('Atom type : ',self.data_sf.keys()) 
        print('Keys      : ' , self.data.keys())
        for name in ['tot_num', 'atom_idx']:
            print('{0}   :   {1}'.format(name , str(self.data[name])))

    #Show atom type & number from OUTCAR
    def show_atom_info(self):
        for name in self.atom_type:
            print('Tpye of atom : {0}  , Number of atoms : {1}'.format(name,self.atom_number[name]))

    def calculate_sf(self, atom,  number, line):
        assert number > 0 and number < self.atom_number[atom]+1, 'Not valid atom number'
        assert line   > 0 and line   < self.length[atom]+1     , 'Not valid parameter line'
        out = 0
        index = 0 # Atom number index
        for i in self.atom_type:
            if i == atom:
                break
            index += self.atom_number[i]
        par_i = self.params_dict[atom][0][line-1]
        par_d = self.params_dict[atom][1][line-1]
        self.distance.set_cutoff(par_d[0])
        if par_i[0] == 2: ## get distance for G2
            distance = self.distance.get_g2_distance(num = number+index, index_i =  par_i[1])
        elif par_i[0] == 4 or par_i[0] == 5: ## get distance for G4 G5
            distance = self.distance.get_g4_distance(num = number+index , index_i = par_i[1] , index_j = par_i[2])
        #Calculate value from SF dictonary
        out = self.symmetry_function_dict[atom][line-1](distance)
        return out


    #Calculating sf   need atom name & atom number
    def calculate_sf_by_atom(self, atom, number):
        assert number > 0 and number < self.atom_number[atom], 'Not valid atom number'
        #Loop for all generated symmetry functions !!
        sf_list = list()
        value = 0  #SF value
        index = 0 # Atom number index
        for i in self.atom_type:
            if i == atom:
                break
            index += self.atom_number[i]
        for num  in range(self.length[atom]):
            ##Get parameter from dictonary
            par_i = self.params_dict[atom][0][num]
            par_d = self.params_dict[atom][1][num]
            #Set cufoff from params
            self.distance.set_cutoff(par_d[0])
            if   par_i[0] == 2: ## get distance for G2
                distance = self.distance.get_g2_distance(num = number+index , index_i = par_i[1])
            elif par_i[0] == 4 or par_i[0] == 5: ## get distance for G4 G5
                distance = self.distance.get_g4_distance(num = number+index , index_i = par_i[1] , index_j = par_i[2])
            #Calculate value from SF dictonary
            value = self.symmetry_function_dict[atom][num](distance)
            sf_list.append(value)
        return np.asarray(sf_list , dtype = np.float64)

    def calculate_sf_by_line(self , atom , line):
        assert line   > 0 and line   < self.length[atom]     , 'Not valid parameter line'
        sf_list = list()
        #Get parameter of SF
        par_i = self.params_dict[atom][0][line-1]
        par_d = self.params_dict[atom][1][line-1]
        self.distance.set_cutoff(par_d[0])
        num = 1
        for i in self.atom_type:
            if i == atom: 
                break
            num += self.atom_number[i]
        for i in range(num,num+self.atom_number[atom]):
            if par_i[0] == 2: ## get distance for G2
                distance = self.distance.get_g2_distance(num = i , index_i =  par_i[1])
            elif par_i[0] == 4 or par_i[0] == 5: ## get distance for G4 G5
                distance = self.distance.get_g4_distance(num = i , index_i = par_i[1] , index_j = par_i[2])
            sf_list.append(self.symmetry_function_dict[atom][line-1](distance))
        return np.asarray(sf_list , dtype = np.float64)

    def get_sf_from_data(self,atom, number, line):
        out = self.data_sf[atom][number-1,line-1]
        return out

    #Get symmetry function values from data file
    def get_sf_from_data_by_line(self, atom, line):
        out = self.data_sf[atom][:,line-1]
        return out

    #Get symmetry function values from data file
    def get_sf_from_data_by_atom(self, atom, number):
        out = self.data_sf[atom][number-1,:]
        return out


