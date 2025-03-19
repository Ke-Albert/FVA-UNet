import os
import time
import datetime

from osgeo import gdal
import numpy as np
from tqdm import tqdm

def modify_value(filepath,out_path,threshold):
    ds = gdal.Open(filepath)  # 打开数据集dataset
    ds_width = ds.RasterXSize  # 获取数据宽度
    ds_height = ds.RasterYSize  # 获取数据高度
    ds_geo = ds.GetGeoTransform()  # 获取仿射地理变换参数
    ds_prj = ds.GetProjection()  # 获取投影信息
    array_band = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    # print(array_band.shape)
    
    # 读取第一个波段全部
    for row in range(0, ds_height):
        # 循环当前波段的行
        for col in range(0, ds_width):
            # 循环当前波段的列
            if array_band[row][col] <threshold:
                # 判断第row行第col列的DN值是否为需要修改的值
                array_band[row][col] = 0
                # 修改该值
            else:
                array_band[row][col]=1
    driver = gdal.GetDriverByName('GTiff')  # 载入数据驱动，用于存储内存中的数组
    ds_result = driver.Create(out_path, ds_width, ds_height, bands=1, eType=gdal.GDT_Float64)
    # 创建一个数组，宽高为原始尺寸
    ds_result.SetGeoTransform(ds_geo)  # 导入仿射地理变换参数
    ds_result.SetProjection(ds_prj)  # 导入投影信息
    ds_result.GetRasterBand(1).SetNoDataValue(-1)  # 将无效值设为-1
    ds_result.GetRasterBand(1).WriteArray(array_band)  # 将结果写入数组
    ds_result.FlushCache()
    ds_result=None
    driver=None
    # 删除内存中的结果，否则结果不会写入图像中

def modify():
    root='D:\\毕设实验数据\\flood_dataset\\sen1flood\\NDVI'
    out_root='D:\\毕设实验数据\\flood_dataset\\sen1flood\\NDVI_mask'
    prefixes=['Bolivia','Ghana','Mekong','Nigeria','Paraguay','Spain','Sri-Lanka','USA']
    # prefixes=['Bolivia']
    prefixes_names=[]
    for prefix in prefixes:
        prefix_names=[i for i in os.listdir(root) if i.startswith(prefix) and i.endswith('.tif')]
        prefixes_names.append(prefix_names)
    for i in range(len(prefixes_names)):
        for j in range(len(prefixes_names[i])):
            prefixes_names[i][j]=os.path.join(root,prefixes_names[i][j])
    
    thresholds=[0.56,0.4,0.43,0.37,0.51,0.3,0.55,0.3]
    print("-----正在进行DN值的修改-----")
    start_time=time.time()
    for i in range(len(prefixes_names)):
        for file_path in prefixes_names[i]:
            modify_value(file_path,os.path.join(out_root,file_path.split('\\')[-1]),thresholds[i])
    # modify_value('D:\毕设实验数据\\flood_dataset\sen1flood\\NDVI\\Bolivia_23014_S1Hand.tif','test',1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("-----完成DN值的修改-----\n用时：{}".format(total_time_str))

def change_value(filepath,out_path):
    ds = gdal.Open(filepath)  # 打开数据集dataset
    ds_width = ds.RasterXSize  # 获取数据宽度
    ds_height = ds.RasterYSize  # 获取数据高度
    ds_geo = ds.GetGeoTransform()  # 获取仿射地理变换参数
    ds_prj = ds.GetProjection()  # 获取投影信息
    array_band = ds.GetRasterBand(1).ReadAsArray().astype(np.float16)
    # print(array_band.shape)

    # mask_25=np.where(array_band==25)
    # array_band[mask_25]=-1
    
    print(array_band)
    mask_0,mask_1=np.where(array_band==0),np.where(array_band==1)
    array_band[mask_0]=1
    array_band[mask_1]=2
    mask_ne1=np.where(array_band==-1)
    array_band[mask_ne1]=1
    driver = gdal.GetDriverByName('GTiff')  # 载入数据驱动，用于存储内存中的数组
    ds_result = driver.Create(out_path, ds_width, ds_height, bands=1, eType=gdal.GDT_Byte)
    # 创建一个数组，宽高为原始尺寸
    ds_result.SetGeoTransform(ds_geo)  # 导入仿射地理变换参数
    ds_result.SetProjection(ds_prj)  # 导入投影信息
    ds_result.GetRasterBand(1).WriteArray(array_band)  # 将结果写入数组
    ds_result.FlushCache()
    ds_result=None
    driver=None
    # 删除内存中的结果，否则结果不会写入图像中
def change():
    # root='D:\\毕设实验数据\\flood_dataset\\sen1flood'
    # for i in ['training','test','valid']:
    #     names=[j for j in os.listdir(os.path.join(root,'DRIVE',i,'ground_true')) if j.endswith('.tif')]
    #     for name in names:
    #         filepath=os.path.join(root,'DRIVE',i,'ground_true',name)
    #         outpath=os.path.join(root,'temp',i,'ground_true',name)
    #         change_value(filepath,outpath)

    root='E:\\毕设实验数据\\flood_dataset\\study area\\temp6'
    names=[i for i in os.listdir(root) if i.endswith('.tif')]
    # names=['7_7.tif']
    for name in tqdm(names):
        filepath=os.path.join(root,name)
        outpath=os.path.join('E:\\毕设实验数据\\flood_dataset\\study area\\temp7',name)
        change_value(filepath,outpath)
    
def main():
    # modify()
    change()
if __name__=='__main__':
    # main()
    # file_name='area2_label_refine.tif'
    # file='E:\\毕设实验数据\\flood_dataset\\chaohu_area_batch\\'+file_name
    # data=gdal.Open(file)
    # width=data.RasterXSize
    # height=data.RasterYSize
    # band=data.GetRasterBand(1)
    # geo=data.GetGeoTransform()
    # prj=data.GetProjection()
    # array=band.ReadAsArray(buf_type=gdal.GDT_Float32)
    # # mask_3=np.where(array==1)
    # output_data=np.full((height+14, width),-1, dtype=np.float32)
    # output_data[:height,:]=array[:,:]
    # # output_data[mask_3]=1
    # driver = gdal.GetDriverByName('GTiff')  # 载入数据驱动，用于存储内存中的数组
    # ds_result = driver.Create('E:\\毕设实验数据\\flood_dataset\\chaohu_area_batch\\chaohu_area2_label.tif', width, height+14, bands=1, eType=gdal.GDT_Float32)
    # # 创建一个数组，宽高为原始尺寸
    # ds_result.SetGeoTransform(geo)  # 导入仿射地理变换参数
    # ds_result.SetProjection(prj)  # 导入投影信息
    # ds_result.GetRasterBand(1).WriteArray(output_data)  # 将结果写入数组
    # ds_result.FlushCache()
    # ds_result=None
    # driver=None
    # data=None

    import pandas as pd

    # 读取Excel文件
    df = pd.read_excel(R'E:\毕设实验数据\气象数据_1990-2020\洞庭湖周边区域_1990-2020.xlsx',sheet_name=0)
    df['DATE'] = pd.to_datetime(df['DATEDAY'].astype(str), format='%Y%m%d')
    # 确保 rain 列是数值类型
    df['tavg'] = pd.to_numeric(df['tavg'], errors='coerce')

    # 删除包含缺失值的行
    df = df.dropna(subset=['tavg'])
    # # # 提取年份作为新的列
    df['YEAR'] = df['DATE'].dt.year
    # df['Month'] = df['DATE'].dt.month
    # annual_mean = df.groupby('YEAR')['r08'].sum().divide(8).reset_index()
    # print(df['YEAR'][2554:2557])
    # month_mean=df.groupby(['Month'])['r08'].sum().divide(7*30).reset_index()
    month_mean=df.groupby(['YEAR'])['tavg'].mean().reset_index()


    print("年平均值：")
    # print(annual_mean['r08'])
    # print(month_mean['r08'])
    print(month_mean['tavg'])