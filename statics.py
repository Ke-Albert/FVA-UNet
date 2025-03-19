import os
import time
import datetime
import shutil

import numpy as np
import pandas as pd
from osgeo import gdal


def statics(img_path):
    data=gdal.Open(img_path)
    width=data.RasterXSize
    height=data.RasterYSize
    data =data.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)#(H,W)
    water=len(np.where(data==1)[0])
    non_water=len(np.where(data==0)[0])
    invalid=len(np.where(data==-1)[0])
    total_pixels=width*height
    with open('static.txt','a') as f:
        text='{},{},{},{},{:.2f}\n'.format(img_path.split('\\')[-1],water,non_water,invalid,water/total_pixels)
        f.write(text)

def main():
    os.chdir('D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\')
    print(os.getcwd())
    with open('./static.txt','a') as f:
        text='影像名,水体像素数量,'+'非水体像素数量,'+'无效像素数量,'+'水体数量百分比\n'
        f.write(text)
    root="D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\Label"
    imgs_path=[os.path.join(root,i) for i in os.listdir(root) if i.endswith('.tif')]
    # print(imgs_path[0])
    for path in imgs_path:
        statics(path)


if __name__=='__main__':
    main()
    