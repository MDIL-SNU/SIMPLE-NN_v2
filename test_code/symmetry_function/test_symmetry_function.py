# Test symmetry function value in OUTCAR or POSCAR to pickled file
import os
import sys
import numpy as np
# All functions in here
import cal_symmetry_function 

def test():
    #for pytest
    root_dir = './test_input/'
    #defalut 
    #root_dir = './'

    outcar  = root_dir+'OUTCAR_GST_comp'
    pickle  = root_dir+'data_comp.pickle'
    pt_data = root_dir+'data_comp.pt'
    yaml    = root_dir+'input_GST.yaml'


    print("TEST symmetry function")
    # Load structure
    structure = cal_symmetry_function.open_outcar(outcar)

    try:
        #Generate Symmetry function using parameters
        #index = 2 : G2  / 4 : G4 / 5 : G5 symmetry function
        print("Set G2 symmetry function with param_d [6.0 0.003214 0.0 0.0]")
        sf_g2 = cal_symmetry_function.generate_sf(index = 2 , param_d = [6.0 , 0.003214 , 0.0 , 0.0])  #[cutoff , eta , R_s , Mone]
        print("Set G4 symmetry function with param_d [6.0 0.003214 4.0 1.0]")
        sf_g4 = cal_symmetry_function.generate_sf(index = 4 , param_d = [6.0 , 0.089277 , 4.0 , 1.0])  #[cutoff , eta , zeta , lambda]
    
        #Get distance information from OUTCAR & POSCAR 
        #Distance_atoms use adjust & get distance from structure
        distance = cal_symmetry_function.Distance_atoms(structure)
        #Set cufoff radius 
        distance.set_cutoff(6.0)
    except:
        print(sys.exc_info())
        raise Exception("Error occured : cannot set symmetry function")
        
    
         
    try:
        print('_'*60)
        print("Validation with G2 symmetry function with pregenerated data")
        cal_list = list() 
        tmp = 0
        for i in range(33,73): 
            tmp = distance.get_g2_distance(i,1)  #  get_g2_dist(atom number , atom type)
            cal_list.append(sf_g2(tmp))  
        cal_list = np.array(cal_list)
        print('Calculated SF :', cal_list)
        pickle_g2 = cal_symmetry_function.load_pickle(pickle)   #From picked data
        pickle_g2 = np.array(pickle_g2['x']['Te'][:,0])
        print('Pickled SF data  :',pickle_g2)
        abs_sum = np.sum(cal_list - pickle_g2)
        assert abs_sum < 1E-5
        print('Absolute sum btw both SF : ',abs_sum)
        print('G2 SF validation OK')
    except AssertionError:
        print(sys.exc_info())
        raise Exception(f"Error occured : large error in g2 symmetry function : {abs_sum} > 1E-5")
    except:
        print(sys.exc_info())
        raise Exception("Error occured : g2 symmetry function wrong")
    
    try:
        print('_'*60)
        print("Validation with G4 symmetry function with pregenerated data")
        cal_list = list()
        tmp = 0
        for i in range(33,73):  
            tmp =  distance.get_g4_distance(i,3,3)
            cal_list.append(sf_g4(tmp))  # get_g4_dist(atom number , atom type_1 , atom type_2)
        print('Testing G4 symmetry function')
        cal_list = np.array(cal_list)
        print('Calculated SF :', cal_list)
        pickle_g4 = cal_symmetry_function.load_pickle(pickle)
        pickle_g4 = np.array(pickle_g4['x']['Te'][:,131])
        print('Pickled SF data  :', pickle_g4)
        abs_sum = np.sum(cal_list - pickle_g4)
        assert abs_sum < 1E-5
        print('Absolute sum btw both SF : ',abs_sum)
        print('G4 SF validation OK')
    except AssertionError:
        print(sys.exc_info())
        raise Exception(f"Error occured : large error in g4 symmetry function : {abs_sum} > 1E-5")
    except:
        print(sys.exc_info())
        raise Exception("Error occured : g4 symmetry function wrong")
    
    
    
    
    ## Use Class to test symmetry function ##
    ## load data of OUTCAR & yaml & pickled data
    load = cal_symmetry_function.Test_symmetry_function(output_name = outcar  , yaml_name = yaml  , data_name = pt_data)
    
    
    print("\nCalculating symmetry function")
    try:
        atype = 'Te'
        idx = np.random.randint(1,41)
        line = np.random.randint(1,133)
        print("Test specific atom of specific SF")
        print(f"Calculate symmetry function of : ( TYPE : {atype} , INDEX : {idx} , PARAM LINE {line})")
        cal = np.array(load.calculate_sf(atom = atype , number = idx , line = line)) 
        pic = np.array(load.get_sf_from_data(atom = atype , number = idx , line = line)) 
        abs_sum = np.abs(cal - pic)
        print('Calculated data  : ',cal)
        print('Pickled    data  : ',pic)
        print('Absulte difference : ', abs_sum)
        assert abs_sum < 1E-4
        print(f"Calculate symmetry function ( TYPE : {atype} , INDEX : {idx} , PARAM LINE {line}) done")
    except AssertionError:
        print(sys.exc_info())
        raise Exception(f"Error occured : SF type of {atype} , {idx}th atom {line}th line wrong value, error {abs_sum}")
    except:
        print(sys.exc_info())
        raise Exception(f"Error occured : cannot calculate symmetry function ")
    
    
    
    try:
        print("\nCalculate all symmetry function of specific atom")
        atype = 'Te'
        idx = np.random.randint(1,40)
        print(f"Calculating all SF of {atype}, {idx} th atom")
        cal = np.array(load.calculate_sf_by_atom(atom =  atype , number = idx))  
        pic = np.array(load.get_sf_from_data_by_atom(atom = atype , number = idx)) 
        abs_sum = np.sum(np.abs(cal-pic))
        print('')
        print('Calculated data  :',cal)
        print('Pickled    data  :',pic)
        assert abs_sum < 5E-1
        print('Absolute sum : ', abs_sum)
    except AssertionError:
        print(sys.exc_info())
        raise Exception(f"Error occured : SF type of {atype} {idx} th atom wrong value, error {abs_num} ")
    except:
        print(sys.exc_info())
        raise Exception(f"Error occured : cannot calculate symmetry function ")
    
    
    
    try:
        print("Calculate symmetry function of all atom for specific line")
        atype = 'Sb'
        line = np.random.randint(1,133)
        print(f"Calculate {line} th SF in param_Sb for all atom of {atype}")
        cal = load.calculate_sf_by_line(atom = atype , line = line)  
        pic = load.get_sf_from_data_by_line(atom = atype , line = line) #same format
        abs_sum = np.sum(np.abs(cal-pic))
        print('')
        print('Calculated data  :',cal)
        print('Pickled    data  :',pic)
        assert abs_sum < 5E-1
        print('Absolute sum : ', abs_sum)
    except AssertionError:
        print(sys.exc_info())
        raise Exception(f"Error occured : SF type of {atype} {line} th line wrong value, error {abs_num} ")
    except:
        print(sys.exc_info())
        raise Exception(f"Error occured : cannot calculate symmetry function ")
     
if __name__ == "__main__":
    test()
