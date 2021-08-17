import os
import sys
#Check ase version available#
import ase


def test():
    #Set yor python directory or command
    python_dir='python3'
    #Set python test files to run
    test_list=[
             'parse.py',
             'tag_weight.py',
             'load_snap.py',
             'save_file.py',
              ]
    
    print('Start testing default function ')
    
    pytest_dir = './test_code/utils/'
    for number, test in enumerate(test_list):
        print(f'TEST {number+1} : {test} start.')
        success_info = eval(r'os.system("{0} {1}")'.format(python_dir,pytest_dir+test))
        if success_info == 0:
            print(f'TEST {number+1} : {test} done.\n')
        else:
            raise Exception(f'TEST {number+1} : in {test} error occured.')
    print("Testing default function done")

if __name__ == "__main__":
    test()
