import sys
import os
sys.path.append('./')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing




def test():
    # Minimum Setting for Testing feature Descriptor class
    #rootdir='./'

    #Set yor python directory or command
    python_dir='python3'
    #Set python test files to run
    test_list=[
    'split_list.py',
    'cal_scale.py',
    'cal_pca.py',
    'prep.py'
    ]
    
    print('Start testing preprocess')

    pytest_dir = './test_code/features/preprocess/'

    for number, test in enumerate(test_list):
        print(f'TEST {number+1} : {test} start.')
        success_info = eval(r'os.system("{0} {1}")'.format(python_dir,pytest_dir+test))
        if success_info == 0:
            print(f'TEST {number+1} : {test} done.\n')
        else:
            raise Exception(f'TEST {number+1} : in {test} error occured.')
    print("Testing preprocess done")
    if os.path.exists('LOG'):
        os.remove('LOG')





if __name__ == "__main__":
    test()

