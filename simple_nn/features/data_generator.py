from braceexpand import braceexpand
import numpy as np
import ase
from ase import io
import os
import torch

""" In this module, functions handles structure list file and OUTCAR files for making data set(format: torch.save)

    parse_structure_list(logfile, structure_list): Parsing "structure_list" file (default="./str_list")
    load_snapshots(inputs, item, logfile): Read structure file and load snapshots using ase.io.read() method
    save_to_datafile(inputs, data, data_idx, logfile): save results to pt files
"""

def parse_structure_list(logfile, structure_list, comm):
    """ Parsing "structure_list" file (default="./structure_list")

    ex) "structure_list" file format:
            [structure_tag1 : weight]
            STRUCTURE_PATH_EXP1 INDEX_EXP1   ex) /PATH/OUTCAR1 ::10
            STRUCTURE_PATH_EXP2 INDEX_EXP2   ex) /PATH/OUTCAR2 10:1000:50
            ...

            [structure_tag2]
            STRUCTURE_PATH_EXP3 INDEX_EXP3   ex) /PATH/OUTCAR{1..10} ::
            STRUCTURE_PATH_EXP4 INDEX_EXP4   ex) /PATH{1..10}/OUTCAR ::
            ...

    1. Seperate structure_tag and weight in between "[ ]"
    2. Extract STRUCTURE_PATH from STRUCTURE_PATH_EXP and INDEX_EXP
    3. Set tag index for each STRUCTURE_PATH
    We support STRUCTURE_PATH_EXP as brace expansion ex) /PATH{1..10}/OUTCAR or /PATH/OUTCAR{1..10}

    Args:
        logfile(file obj): logfile object
        structure_list(str): name of structure list file that we have to parse
    Returns:
        structure_tags(str list): list of structure tag
        structure_weights(float list): list of structure weight
        structure_file_list(str list): list of STRUCTURE_PATH
        structure_slicing_list(str list): list of INDEX_EXP
        structure_tag_idx(int list): list of structure tag index of each STRUCTURE_PATH
    """
    structure_file_list = []
    structure_slicing_list = []
    structure_tag_idx = []
    structure_tags = ['None']
    structure_weights = [1.0]
    tag = 'None'
    weight = 1.0

    with open(structure_list, 'r') as fil:
        for line in fil:
            line = line.strip()

            if len(line) == 0 or line.isspace():
                continue
            # 1. Extract structure tag and weight in between "[ ]"
            elif line[0] == '[' and line[-1] == ']':
                tag_line = line[1:-1]
                tag, weight = _get_tag_and_weight(tag_line)

                if weight < 0:
                    err = "Structure weight must be greater than or equal to zero."
                    if comm.rank == 0:
                        logfile.write("Error: {:}\n".format(err))
                    raise ValueError(err)
                elif np.isclose(weight, 0):
                    if comm.rank == 0:
                        logfile.write("Warning: Structure weight for '{:}' is set to zero.\n".format(tag))

                # If the same structure tags are given multiple times with different weights,
                # other than first value will be ignored!
                # Validate structure weight (structure weight is not validated on training run).
                if tag not in structure_tags:
                    structure_tags.append(tag)
                    structure_weights.append(weight)
                else:
                    existent_weight = structure_weights[structure_tags.index(tag)]
                    if comm.rank == 0:
                        if not np.isclose(existent_weight - weight, 0):
                            logfile.write("Warning: Structure weight for '{:}' is set to {:} (previously set to {:}). New value will be ignored\n"\
                                                    .format(tag, weight, existent_weight))
                continue
            # 2. Extract STRUCTURE_PATH and INDEX, then set structure tag index
            else:
                tmp_split = line.split()
                try:
                    structure_file_exp = tmp_split[0]
                    structure_slicing = tmp_split[1]
                    for structure_file in list(braceexpand(structure_file_exp)):
                        structure_file_list.append(structure_file)
                        structure_slicing_list.append(structure_slicing)
                        structure_tag_idx.append(structure_tags.index(tag))
                except:
                    err = "Unexpected line format in [str_list]"
                    if comm.rank == 0:
                        logfile.write("\nError: {:}\n".format(err))
                        logfile.write("ERROR LINE: {:}\n".format(line))
                    raise NotImplementedError(err)

    return structure_tags, structure_weights, structure_file_list, structure_slicing_list, structure_tag_idx

# Structure tag is in front of ":" and weight is back of ":" ex) [structur_tag : weight]
# If no explicit weight(no ":"), default weight=1.0  ex) [structure_tag]
def _get_tag_and_weight(text):
    if ':' in text:
        splited_text = text.rsplit(':', 1)
        try:
            tag = splited_text[0].strip()
            weight = float(splited_text[1].strip())
        except ValueError:
            tag = text.strip()
            weight = 1.0
    else:
        tag = text.strip()
        weight = 1.0

    return tag, weight

def load_structures(inputs, structure_file, structure_slicing, logfile, comm):
    """ Read structure file and load structures using ase.io.read() method

    Handle inputs['refdata_format']: 'vasp-out' vs else
    Handle inputs['compress_outcar']: True vs False
    Handle ase version: later than 3.18.0 vs else

    Args:
        inputs(dict): ['descriptor'] part in input.yaml
        structure_file(str): structure file path
        structure_slicing(str): index expression    ex) "::10", "8"
        logfile(file obj): logfile object
    Returns:
        structures(ase.atoms.Atoms object): Atoms object from ase module that contain structure information, E, F, S ...
    """
    file_path = structure_file
    if ':' in structure_slicing:
        index = structure_slicing
    else:
        index = int(structure_slicing)

    if comm.rank == 0:
        logfile.write(f"{file_path} {index}")

    if inputs['descriptor']['refdata_format'] == 'vasp-out':
        if inputs['descriptor']['compress_outcar']:
            if comm.rank == 0:
                file_path = compress_outcar(structure_file)
            file_path = comm.bcast(file_path, root=0);

        if ase.__version__ >= '3.18.0':
            structures = io.read(file_path, index=index, format=inputs['descriptor']['refdata_format'], parallel=False)
        else:
            structures = io.read(file_path, index=index, format=inputs['descriptor']['refdata_format'], force_consistent=True, parallel=False)
    else:
        if comm.rank == 0:
            logfile.write("Warning: Structure format is not OUTCAR(['refdata_format'] : {:}). Unexpected error can occur.\n"\
                                                    .format(inputs['descriptor']['refdata_format']))
        structures = io.read(file_path, index=index, format=inputs['descriptor']['refdata_format'], parallel=False)

    return structures

def save_to_datafile(inputs, data, data_idx, logfile):
    """ Write result data to pt file

    Check if pt list file is open
    Save data as data{index}.pt ({index}: +1 of last index in save directory)
    Write pt file path in pt list

    Args:
        inputs(dict): ['descriptor'] part in input.yaml
        data(dic): result data that contains information after calculting symmetry function values
        data_idx(int): index of data file to save
        logfile(file obj): logfile object
    Returns:
        tmp_filename(str): saved pt file path
    """
    data_dir = inputs['descriptor']['save_directory']

    try:
        tmp_filename = os.path.join(os.getcwd() if inputs['descriptor']['absolute_path'] else '', data_dir, 'data{}.pt'.format(data_idx))
        torch.save(data, tmp_filename)
    except:
        err = "Unexpected error during save data to .pt file"
        if comm.rank == 0:
            logfile.write("\nError: {:}\n".format(err))
        raise NotImplementedError(err)

    return tmp_filename

def compress_outcar(filename):
    """
    Compress VASP OUTCAR file for fast file-reading in ASE.
    Compressed file (tmp_comp_OUTCAR) is temporarily created in the current directory.

    :param str filename: filename of OUTCAR

    supported properties:

    - atom types
    - lattice vector(cell)
    - free energy
    - force
    - stress
    """
    comp_name = './tmp_comp_OUTCAR'

    with open(filename, 'r') as fil, open(comp_name, 'w') as res:
        minus_tag = 0
        line_tag = 0
        ions_key = 0
        for line in fil:
            if 'POTCAR:' in line:
                res.write(line)
            if 'POSCAR:' in line:
                res.write(line)
            elif 'ions per type' in line and ions_key == 0:
                res.write(line)
                ions_key = 1
            elif 'direct lattice vectors' in line:
                res.write(line)
                minus_tag = 3
            elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                res.write(line)
                minus_tag = 4
            elif 'POSITION          ' in line:
                res.write(line)
                line_tag = 3
            elif 'FORCE on cell =-STRESS' in line:
                res.write(line)
                minus_tag = 15
            elif 'Iteration' in line:
                res.write(line)
            elif minus_tag > 0:
                res.write(line)
                minus_tag -= 1
            elif line_tag > 0:
                res.write(line)
                if '-------------------' in line:
                    line_tag -= 1

    return comp_name

