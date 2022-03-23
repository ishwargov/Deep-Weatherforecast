import os
import gdown

id = '1cUcfD1u8_QWGXs8204k2r5_uihGbQMqx'
output = 'dat.zip'
gdown.download(id = id,output=output,quiet=False)
os.system('unzip ./dat.zip')
os.system('rm dat.zip')