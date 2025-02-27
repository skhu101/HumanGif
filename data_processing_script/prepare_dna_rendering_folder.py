import os
import shutil

test_lst = ['0031_03', '0034_04', '0094_02', '0307_03']
with open('data/DNA_Rendering/Part_1/smc_lst.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line not in test_lst:
            src = 'data/DNA_Rendering/Part_1/data_render/' + line
            dst = 'data/DNA_Rendering/train/' + f'Part_1_{line}'
        else:
            src = 'data/DNA_Rendering/Part_1/data_render/' + line
            dst = 'data/DNA_Rendering/test/' + f'Part_1_{line}'
        os.symlink(src, dst)

test_lst = ['0007_07', '0016_01', '0019_09', '0044_07', '0078_11', '0128_12']
with open('data/DNA_Rendering/Part_1/smc_lst.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line not in test_lst:
            src = 'data/DNA_Rendering/Part_1/data_render/' + line
            dst = 'data/DNA_Rendering/train/' + f'Part_1_{line}'
        else:
            src = 'data/DNA_Rendering/Part_1/data_render/' + line
            dst = 'data/DNA_Rendering/test/' + f'Part_1_{line}'
        os.symlink(src, dst)

test_lst = ['0031_03', '0034_04', '0094_02', '0307_03']
with open('data/DNA_Rendering/Part_2/smc_lst.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line not in test_lst:
            src = 'data/DNA_Rendering/Part_2/data_render/' + line
            dst = 'data/DNA_Rendering/train/' + f'Part_2_{line}'
        else:
            src = 'data/DNA_Rendering/Part_2/data_render/' + line
            dst = 'data/DNA_Rendering/test/' + f'Part_2_{line}'
        os.symlink(src, dst)

test_lst = ['0007_07', '0016_01', '0019_09', '0044_07', '0078_11', '0128_12']
with open('data/DNA_Rendering/Part_2/smc_lst.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line not in test_lst:
            src = 'data/DNA_Rendering/Part_2/data_render/' + line
            dst = 'data/DNA_Rendering/train/' + f'Part_2_{line}'
        else:
            src = 'data/DNA_Rendering/Part_2/data_render/' + line
            dst = 'data/DNA_Rendering/test/' + f'Part_2_{line}'
        os.symlink(src, dst)
