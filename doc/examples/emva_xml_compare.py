import os
from emva1288 import process

# specify the path to the files to compare
dir_ = '/home/work/1288/datasets/EMVA1288_ReferenceSet_001_CCD_12Bit/'
fresult1 = 'EMVA1288_Result1.xml'
fresult2 = 'EMVA1288_Result2.xml'
fcompare = 'EMVA1288_Compare.txt'

u = process.routines.compare_xml(os.path.join(dir_, fresult1),
                                 os.path.join(dir_, fresult2),
                                 os.path.join(dir_, fcompare))
print(u)
