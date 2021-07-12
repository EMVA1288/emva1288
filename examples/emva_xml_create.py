import os
from emva1288 import process

# specify the path to the image stack
dir_ = '/home/work/1288/datasets/EMVA1288_ReferenceSet_001_CCD_12Bit/'
fname = 'EMVA1288_Data.txt'
fname = os.path.join(dir_, fname)
fresult = 'EMVA1288_Result.xml'
fresult = os.path.join(dir_, fresult)

parser = process.ParseEmvaDescriptorFile(os.path.join(dir_, fname))
imgs = process.LoadImageData(parser.images)
dat = process.Data1288(imgs.data)

res = process.Results1288(dat.data)

res.print_results()
f = open(os.path.join(dir_, fresult), "wb")
f.write(res.xml())
f.close()
