import copy
import os
import time
import datetime
import shutil

import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm

def merge_band():
    # names=[i for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\chaohu_area_batch\\sia') if i.endswith('.tif')]
    # names=['Bolivia_76104_S1Hand.tif']
    names=['chaohu_area2.tif']
    count=0
    step=len(names)//10
    for name in tqdm(names):
        # count+=1
        # if count%step==0:
        #     print('目前进度{:.1f}%'.format(count/len(names)*100))
        file1=os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\intensity','0718.tif')
        file2=os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\ic','0507_0718.tif')
        file3=os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\ic','0425_0507.tif')
        file4=os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\intensity','0507.tif')
        out_path=os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\images','0718.tif')

        merge1 = gdal.Open(file1)  # 打开数据集dataset
        merge2=gdal.Open(file2)
        merge3 = gdal.Open(file3)
        merge4 = gdal.Open(file4)
        width=merge1.RasterXSize
        height=merge2.RasterYSize
        geo=merge1.GetGeoTransform()
        prj=merge1.GetProjection()

        merge1_bands=merge1.RasterCount
        merge2_bands=merge2.RasterCount
        merge3_bands=merge3.RasterCount
        merge4_bands=merge4.RasterCount
    
        # transform to np
        for i in range(merge1_bands):
            data = merge1.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
            data = np.expand_dims(data, 2)
            if i == 0:
                img_matrix_1 = data
            else:
                img_matrix_1 = np.concatenate((img_matrix_1, data), axis=2)
        for i in range(merge2_bands):
            data = merge2.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
            data = np.expand_dims(data, 2)
            if i == 0:
                img_matrix_2 = data
            else:
                img_matrix_2 = np.concatenate((img_matrix_2, data), axis=2)
        for i in range(merge3_bands):
            data = merge3.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
            data = np.expand_dims(data, 2)
            if i == 0:
                img_matrix_3 = data
            else:
                img_matrix_3 = np.concatenate((img_matrix_3, data), axis=2)
        for i in range(merge4_bands):
            data = merge4.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
            data = np.expand_dims(data, 2)
            if i == 0:
                img_matrix_4 = data
            else:
                img_matrix_4 = np.concatenate((img_matrix_4, data), axis=2)
        img_matrix=np.concatenate((img_matrix_1,img_matrix_2,
                                   img_matrix_3,img_matrix_4
                                   ),axis=2)
        

        
        # zero_mask=(np.where(img_matrix[:,:,0]==0) or np.where(img_matrix[:,:,2]==0) )
        # # or np.where(img_matrix[:,:,4]==0) or np.where(img_matrix[:,:,6]==0)
        # img_matrix[zero_mask]=np.nan

        # #将NaN的区域赋值为整张图像的均值
        for i in range(merge1_bands+merge2_bands+merge3_bands+merge4_bands):
            nan_mask=np.isnan(img_matrix[:,:,i])
            # mean_val=np.nanmean(img_matrix[:,:,i])
            mean_val=0
            img_matrix[:,:,i][nan_mask]=mean_val
        
        #写入磁盘
        driver = gdal.GetDriverByName('GTiff')  # 载入数据驱动，用于存储内存中的数组
        dataset = driver.Create(out_path, width, height, bands=merge1_bands+merge2_bands
                                 +merge3_bands+merge4_bands
                                 , eType=gdal.GDT_Float32)
      
        # 创建一个数组，宽高为原始尺寸
        dataset.SetGeoTransform(geo)  # 导入仿射地理变换参数
        dataset.SetProjection(prj)  # 导入投影信息
        for i in range(merge1_bands+merge2_bands
                       +merge3_bands+merge4_bands
                       ):
            dataset.GetRasterBand(i + 1).WriteArray(img_matrix[:, :, i])
        dataset.FlushCache()
        dataset=None
        driver=None

def remove():
    # names1=[i for i in os.listdir('D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\Label_filtered') if i.endswith('.tif')]
    # names2=[i for i in os.listdir('D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\NDVI') if i.endswith('.tif')]
    # names=list(set(names2)-set(names1))
    # for name in names:
    #     file_path=os.path.join('D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\NDVI',name)
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #         print(f'文件{name}已被删除')
    #     else:
    #         print(f"文件{name}不存在")


    # pres='Mekong_'
    # names=[pres+str(i)+'_S1OtsuLabelWeak.tif' for i in [5606220,334562,2544438,2013456,3597833,6621401,2094780,429512,6904489,5892412,9325080,6024020,1685525,5806305,5368035,8992647,8464155,6377119,5533064,2615980,1879453]]
    # for name in names:
    #     file_path=os.path.join('D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\Label_filtered',name)
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    #         print(f'文件{name}已被删除')
    #     else:
    #         print(f"文件{name}不存在")
    path='D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\co_IC\\temp1'
    temps=[i for i in os.listdir(path) if i.endswith('.tif')]
    for temp in temps:
        data=gdal.Open(os.path.join(path,temp))
        img = data.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)
        zero=len(np.where(img==0)[0])
        negetive=len(np.where(img==-1)[0])
        del data,img
        if zero>=400*400 or negetive>=300*300:
            if os.path.exists(os.path.join(path,temp)):
                os.remove(os.path.join(path,temp))
                print(f'文件{temp}已被删除')
            else:
                print(f"文件{temp}不存在")

def begin_merge():
    print('开始融合')
    start_time=time.time()
    merge_band()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('融合结束，用时: {}'.format(total_time_str))

def move():
    """
    用于移动影像数据到其他的存储位置，首先读取需要移动数据的csv文件，之后根据读取的影响名称找到数据位置进行移动
    """
    root='D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\水体数据统计.xlsx'
    source_root='D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\S1'
    target_root='D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+\\S1_filtered'
    train='flood_train_data.csv'
    valid='flood_valid_data.csv'
    test='flood_test_data.csv'

    
    # for i in [train,valid,test]:
    #     df=pd.read_csv(os.path.join(root,i))
    #     for _,row in df.iterrows():
    #         x,y=row
    #         # print(x,y)
    #         # break
    #         temp=i.split('.')[0].split('_')[1]
    #         # shutil.copy(os.path.join(source_root,'S1Hand_IC',x),os.path.join(target_root,temp,'images',x))#images
    #         # shutil.copy(os.path.join(source_root,'NDVI_mask',x),os.path.join(target_root,temp,'mask',x))#mask
    #         shutil.copy(os.path.join(source_root,'LabelHand',y),os.path.join(target_root,temp,'ground_true',y))#ground true

    # xls=pd.ExcelFile(root)
    # df = pd.read_excel(xls, sheet_name='coarse_filted_data')
    # for _,row in df.iterrows():
    #     # Bolivia_5631499_S1OtsuLabelWeak.tif [226412, 35732, 0, 0.86]
    #     x,*y=row
    #     # shutil.copy(os.path.join(source_root,x),os.path.join(target_root,x))
    #     # Bolivia_5631499_S1Weak.tif
    #     temp='_'.join(x.split('_')[:-1])+'_S1Weak.tif'
    #     shutil.copy(os.path.join(source_root,temp),os.path.join(target_root,temp))
    txt_root='E:\\毕设实验数据\\flood_dataset\\DRIVE\\test.txt'
    prefix='test'
    with open(txt_root,'r') as f:
        data=f.read()
    number_list=data.split('\n')[:-1]
    img_names_list=set([i for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\S1_IC') if i.endswith('.tif')])
    names=['Bolivia_','Ghana_','Mekong_','Nigeria_','Paraguay_','Spain_','Sri-Lanka_','USA_']
    filtered_list=[]
    for name in names:
        temp=set([name+i+'.tif' for i in number_list])
        filtered_list+=list(temp & img_names_list)

    for i in filtered_list:
        #i='Bolivia_312675.tif'
        shutil.copy(os.path.join('E:\\毕设实验数据\\flood_dataset\\S1_IC',i),os.path.join('E:\\毕设实验数据\\flood_dataset\\DRIVE\\'+prefix+'\\images',i))
        shutil.copy(os.path.join('E:\\毕设实验数据\\flood_dataset\\MASK',i.split('.')[0]+'_mask.tif'),os.path.join('E:\\毕设实验数据\\flood_dataset\\DRIVE\\'+prefix+'\\mask',i.split('.')[0]+'_mask.tif'))
        shutil.copy(os.path.join('E:\\毕设实验数据\\flood_dataset\\Label',i.split('.')[0]+'_label.tif'),os.path.join('E:\\毕设实验数据\\flood_dataset\\DRIVE\\'+prefix+'\\ground_true',i.split('.')[0]+'_label.tif'))

def one_band_statics(one_band):
    """
    用于统计单波段影像的统计量，包括均值，方差，标准差，最小值，最大值
    """
    result=0.
    result=sum(one_band)
    return result

def one_image_statics(img_path):
    gdal_data=gdal.Open(img_path)
    bands_num=gdal_data.RasterCount
    # bands_sum=[0 for _ in range(bands_num)]
    bands_std_sum=[0 for _ in range(bands_num)]
    mean=[-13.15557343,-21.45834872,0.27024694,0.28602373,0.27478102,0.29397414,-12.55040084,-19.76063181]
    h=gdal_data.RasterYSize
    w=gdal_data.RasterXSize
    for i in range(bands_num):
        one_band=np.array(gdal_data.GetRasterBand(i+1).ReadAsArray(buf_type=gdal.GDT_Float32))
        # bands_sum[i]=np.sum(one_band)
        bands_std_sum[i]=np.sum(np.subtract(one_band,mean[i])**2)
    return bands_std_sum,h*w

def check_nan(img_path):
    """
    用于检查影像中是否存在nan值
    """
    gdal_data=gdal.Open(img_path)
    bands_num=gdal_data.RasterCount
    for i in range(bands_num):
        one_band=np.array(gdal_data.GetRasterBand(i+1).ReadAsArray(buf_type=gdal.GDT_Float32))
        nan_num=np.count_nonzero(np.isnan(one_band))
        if nan_num>0:
            break
    del gdal_data
    return nan_num

def datasets_statics():
    # datasets_bands_mean=np.array([float(0) for _ in range(8)])
    datasets_bands_std=np.array([float(0) for _ in range(8)])
    datasets_pixels_num=0
    for i in [_ for _ in os.listdir('E:\\毕设实验数据\\flood_dataset\\S1_IC') if _.endswith('.tif')]:
        img_path=os.path.join('E:\\毕设实验数据\\flood_dataset\\S1_IC',i)
        bands_std_sum,pixels_sum=one_image_statics(img_path)
        datasets_bands_std+=np.array(bands_std_sum)
        datasets_pixels_num+=pixels_sum
    print('std:{}'.format((datasets_bands_std/datasets_pixels_num)**0.5))
    print('pixels_num:{}'.format(datasets_pixels_num))
def main():
    begin_merge()
if __name__=='__main__':
    # main()
    names=['0718_clipped.tif','0730_clipped.tif','0811_clipped.tif','0823_clipped.tif','0904_clipped.tif']

    for name in names:
        path=os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\predict_tif',name)
        file1=gdal.Open(path)
        file2=gdal.Open(R'E:\毕设实验数据\flood_dataset\study area\dong\dong_water_mask_clipped.tif')
        img1=file1.ReadAsArray(buf_type=gdal.GDT_Byte)
        img2=file2.ReadAsArray(buf_type=gdal.GDT_Byte)

        mask=np.where(img2==0)
        img1[mask]=0
        img1=np.expand_dims(img1, 2)
        width=file1.RasterXSize
        height=file1.RasterYSize
        geo=file1.GetGeoTransform()
        prj=file1.GetProjection()
        
        #写入磁盘
        driver = gdal.GetDriverByName('GTiff')  # 载入数据驱动，用于存储内存中的数组
        
        dataset = driver.Create(os.path.join(R'E:\毕设实验数据\flood_dataset\study area\dong\predict_tif','mask_'+name), 
                                width, height, bands=1
                                 , eType=gdal.GDT_Byte)
        
        # 创建一个数组，宽高为原始尺寸
        dataset.SetGeoTransform(geo)  # 导入仿射地理变换参数
        dataset.SetProjection(prj)  # 导入投影信息
        dataset.GetRasterBand(1).WriteArray(img1[:, :, 0])
        dataset.FlushCache()
        dataset=None
        driver=None