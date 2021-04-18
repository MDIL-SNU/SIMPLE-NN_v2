from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='view set transformer training data')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-f', '--force', action='store_true')
parser.add_argument('-m', '--multiplot', action='store_true')
parser.add_argument('-n', '--noshow', action='store_true')
parser.add_argument('--epochs', default=-1, type=int)
parser.add_argument('--interval', default=1, type=int)
parser.add_argument('--compare', action='store_true')
args = parser.parse_args()

if args.multiplot == True:
    args.force = True

if args.compare:
    args.data = open(args.data)
    color_divider = 2
else:
    args.data = [[args.data, '']]
    color_divider = 1

is_force = False
max_epoch = None

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
linestyles = ['-', ':']

if not args.noshow:
    if args.multiplot:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

        ax1.axhline(y=5e-3, linestyle=':', linewidth=0.7, color='k', label='Target err.')
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE$_{\mathrm{E}}$ (eV/atom)')

        ax2.axhline(y=1e-1, linestyle=':', linewidth=0.7, color='k', label='Target err.')

        ax2.set_yscale('log')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE$_{\mathrm{F}}$ (eV/${\mathrm{\AA}}$)')
    else:
        fig, (ax) = plt.subplots(1, 1, sharey=False)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')

        if args.force:
            ax.axhline(y=1e-1, linestyle=':', linewidth=0.7, color='k', label='Target err.')
            ax.set_ylabel('RMSE$_{\mathrm{F}}$ (eV/${\mathrm{\AA}}$)')
        else:
            ax.axhline(y=5e-3, linestyle=':', linewidth=0.7, color='k', label='Target err.')
            ax.set_ylabel('RMSE$_{\mathrm{E}}$ (eV/atom)')
        
color_idx = 0
for filepath in args.data:
    #outputs = glob(f'{args.data}')
    if args.compare:
        filepath = filepath.split(',')
    tmp_filepath, legend_head = filepath
    tmp_filepath = tmp_filepath.replace('\n','')
    legend_head = legend_head.replace('\n','')
    outputs = glob(f"{tmp_filepath}")
    outputs.sort()

    train_eloss = list()
    train_floss = list()

    valid_eloss = list()
    valid_floss = list()
    valid_loss = list()

    train_epoch = 0
    valid_epoch = 0

    #if len(outputs) == 1:
    #if True:

    train_label = legend_head if args.compare else 'Train'
    valid_label = None if args.compare else 'Valid'

    for output in outputs:
        print(output)
        with open(output) as fil:
            train_tag = ['Epoch (GPU:0)']
            valid_tag = ['Valid (GPU:0)']
            for line in fil:
                if len(train_tag) < 2 and train_tag[0] in line:
                    iter_train_max = int(line.split('/')[1].split(']')[0])
                    train_tag.append(f'[{iter_train_max-1}/{iter_train_max}]')
                    print(train_tag)

                if len(valid_tag) < 2 and valid_tag[0] in line:
                    iter_valid_max = int(line.split('/')[1].split(']')[0])
                    valid_tag.append(f'[{iter_valid_max-1}/{iter_valid_max}]')
                    print(valid_tag)

                if train_tag[0] in line and train_tag[1] in line:
                    if args.epochs < 0 or args.epochs > int(line.split('[')[1].split(']')[0]):
                        train_eloss.append(float(line.split()[22]))
                        if args.force:
                            train_floss.append(float(line.split()[28]))
                elif valid_tag[0] in line and valid_tag[1] in line:
                    if args.epochs < 0 or args.epochs > int(line.split('[')[1].split(']')[0]):
                        valid_loss.append(float(line.split()[16]))
                        valid_eloss.append(float(line.split()[22]))
                        if args.force:
                            valid_floss.append(float(line.split()[28]))
                        

            while len(train_eloss) > len(valid_eloss):
                train_eloss.pop()

            while len(valid_eloss) > len(train_eloss):
                valid_eloss.pop()

            train_epoch = len(train_eloss)
            valid_epoch = len(valid_eloss)

            print(f'Train epoch: {train_epoch}, Valid epoch: {valid_epoch}')

    min_idx = np.argmin(valid_loss)
    print(min_idx)

    if args.multiplot:

        if not args.noshow:
            ax1.plot(list(range(len(train_eloss)))[::args.interval], train_eloss[::args.interval], label=train_label, linewidth=0.8, 
                     color=colors[color_idx//color_divider], linestyle=linestyles[color_idx%color_divider])
            ax1.plot(list(range(len(valid_eloss)))[::args.interval], valid_eloss[::args.interval], label=valid_label, linewidth=0.8, 
                     color=colors[(color_idx+1)//color_divider], linestyle=linestyles[(color_idx+1)%color_divider])

            ax2.plot(list(range(len(train_floss)))[::args.interval], train_floss[::args.interval], label=train_label, linewidth=0.8, 
                     color=colors[color_idx//color_divider], linestyle=linestyles[color_idx%color_divider])
            ax2.plot(list(range(len(valid_floss)))[::args.interval], valid_floss[::args.interval], label=valid_label, linewidth=0.8, 
                     color=colors[(color_idx+1)//color_divider], linestyle=linestyles[(color_idx+1)%color_divider])
            
        
        #print(min(train_eloss), min(valid_eloss))
        #print(min(train_floss), min(valid_floss))
        print(train_eloss[min_idx], valid_eloss[min_idx])
        print(train_floss[min_idx], valid_floss[min_idx])

    else:
        if not args.noshow:

            if args.force:
                ax.plot(list(range(len(train_floss)))[::args.interval], train_floss[::args.interval], label=train_label, linewidth=0.8, 
                        color=colors[color_idx//color_divider], linestyle=linestyles[color_idx%color_divider])
                ax.plot(list(range(len(valid_floss)))[::args.interval], valid_floss[::args.interval], label=valid_label, linewidth=0.8, 
                        color=colors[(color_idx+1)//color_divider], linestyle=linestyles[(color_idx+1)%color_divider])
            else:
                ax.plot(list(range(len(train_eloss)))[::args.interval], train_eloss[::args.interval], label=train_label, linewidth=0.8,
                        color=colors[color_idx//color_divider], linestyle=linestyles[color_idx%color_divider])
                ax.plot(list(range(len(valid_eloss)))[::args.interval], valid_eloss[::args.interval], label=valid_label, linewidth=0.8, 
                        color=colors[(color_idx+1)//color_divider], linestyle=linestyles[(color_idx+1)%color_divider])

        if args.force:
            print(train_floss[min_idx], valid_floss[min_idx])
        else:
            print(train_eloss[min_idx], valid_eloss[min_idx])

    color_idx += 2

if not args.noshow:
    if args.multiplot:
        ax1.legend()
        ax2.legend()

    else:
        ax.legend()

    plt.tight_layout()
    plt.show()

if args.compare:
    args.data.close()
