import os
import argparse
import shutil
import time
import datetime
import json
import random

import numpy as np
from osgeo import gdal, ogr, osr
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from PIL import Image


# reading TIFF image
class RASTER:
    @staticmethod
    def open(path):
        dataset = gdal.Open(path)
        return dataset

    @staticmethod
    def write_image(fn_out, im_data, transform, proj=None):
        # 设置投影，proj为wkt format
        if proj is None:
            proj = 'GEOGCS["WGS 84",\
                        DATUM["WGS_1984",\
                                SPHEROID["WGS 84",6378137,298.257223563, \
                                        AUTHORITY["EPSG","7030"]], \
                                AUTHORITY["EPSG","6326"]], \
                        PRIMEM["Greenwich",0, \
                                AUTHORITY["EPSG","8901"]], \
                        UNIT["degree",0.0174532925199433, \
                                AUTHORITY["EPSG","9122"]],\
                        AUTHORITY["EPSG","4326"]]'
        # 渲染为rgba矩阵
        # 设置数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_Int16
        else:
            datatype = gdal.GDT_Float32
        # 将(通道数、高、宽)顺序调整为(高、宽、通道数)
        # print('shape of im data:', im_data.shape)
        im_bands = min(im_data.shape)
        im_shape = list(im_data.shape)
        im_shape.remove(im_bands)
        im_height, im_width = im_shape
        band_idx = im_data.shape.index(im_bands)
        # 找出波段是在第几个

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(fn_out, im_width, im_height, im_bands, eType=datatype)

        # if dataset is not None:
        dataset.SetGeoTransform(transform)  # 写入仿射变换参数
        dataset.SetProjection(proj)  # 写入投影

        if im_bands == 1:

            # print(im_data[:, 0,:].shape)
            if band_idx == 0:
                dataset.GetRasterBand(1).WriteArray(im_data[0, :, :])
            elif band_idx == 1:
                dataset.GetRasterBand(1).WriteArray(im_data[:, 0, :])
            elif band_idx == 2:
                dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])

        else:

            for i in range(im_bands):
                if band_idx == 0:
                    dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
                elif band_idx == 1:
                    dataset.GetRasterBand(i + 1).WriteArray(im_data[:, i, :])
                elif band_idx == 2:
                    dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])

        dataset.FlushCache()
        dataset = None
        driver = None

    @staticmethod
    def clip_roi(data, roi, out_put_name='test', dstpath='E:\\毕设实验数据\\flood_dataset\\study area\\temp60619_db\\'):
        # # 获取ROI的几何信息（边界）
        roi_geotransform = roi.GetGeoTransform()
        width = roi.RasterXSize
        height = roi.RasterYSize
        minX = roi_geotransform[0]
        minY = roi_geotransform[3]
        xRes = roi_geotransform[1]
        yRes = roi_geotransform[5]
        maxX = minX + width * xRes
        maxY = minY + height * yRes
        proj = roi.GetProjection()

        data_geotransform = data.GetGeoTransform()
        data_width = data.RasterXSize
        data_height = data.RasterYSize
        data_minx = data_geotransform[0]
        data_miny = data_geotransform[3]
        data_maxx = data_minx + data_width * xRes
        data_maxy = data_miny + data_height * yRes

        bands = data.RasterCount

        # print(f'roi区域的范围：({minX},{minY}),({maxX},{maxY})')
        # print(f'数据区域范围：({data_minx},{data_miny}),({data_maxx},{data_maxy})')
        # 判定roi是否在数据区域内,严格来说在相等的时候只有当roi的大小为一个像素时才有可能裁剪出来，所以这里取>=几乎不会出错
        # 左上角的(minx,miny)和右下角的(maxx,maxy)显然具有误导性，其实miny对应的是y坐标的最大值，而maxy对应的是y坐标的最小值，因为南北方向的分辨率通常取负数,这样使得影像的扩张方向为右下方
        # ^y
        # |
        # |
        # ---------------------->x
        #                |
        # |

        if data_maxy >= minY or data_miny <= maxY or data_maxx <= minX or data_minx >= maxX:
            del data
            del roi
            print('跳过')
            return
        else:
            # 转换成np格式
            img_matrix = None
            for i in range(bands):
                img = data.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
                img = np.expand_dims(img, 2)
                if i == 0:
                    img_matrix = img
                else:
                    img_matrix = np.concatenate((img_matrix, img), axis=2)
            # 没有溢出的情况
            if minX >= data_minx and maxX <= data_maxx and minY <= data_miny and maxY >= data_maxy:
                x_begin = int((minX - data_minx) // xRes)
                y_begin = int(abs((data_miny - minY) // yRes))
                output_data = img_matrix[y_begin:y_begin + height, x_begin:x_begin + width, :]
            else:
                # 溢出了的情况
                x_bias = y_bias = 0
                x_bias2, y_bias2 = width, height
                # 判断左上角
                x_begin = int((minX - data_minx) // xRes)
                x_end = x_begin + width
                y_begin = int((minY - data_miny) // yRes)
                y_end = y_begin + height

                if x_begin < 0:
                    x_bias = -x_begin
                if y_begin < 0:
                    y_bias = -y_begin
                # 判断右下角
                if x_end > data_width:
                    x_bias2 -= (x_end - data_width)
                if y_end > data_height:
                    y_bias2 -= (y_end - data_height)
                # 创建一个-1矩阵(Height,Width,Bands),因为水体的IC值为0，所以这里默认使用-1来代替无值区域
                output_data = np.full((height, width, bands), -1, dtype=np.float32)
                print(y_bias, y_bias2, x_bias, x_bias2, x_begin, x_end, y_begin, y_end, data_width, data_height)
                output_data[y_bias:y_bias2, x_bias:x_bias2:] = img_matrix[y_begin + y_bias:y_end + y_bias2 - height,
                                                               x_begin + x_bias:x_end + x_bias2 - width, :]
            output_transform = roi_geotransform
            RASTER.write_image(dstpath + out_put_name + '.tif', output_data, output_transform, proj)

        # 清理，关闭数据集
        del data
        del roi
        del output_data
        del img_matrix

    @classmethod
    def split_image(cls, fn_out, origin_data, origin_transform, output_size, proj):
        origin_size = origin_data.shape
        x = origin_transform[0]
        y = origin_transform[3]
        x_step = origin_transform[1]
        y_step = origin_transform[5]
        output_x_step = x_step
        output_y_step = y_step
        for i in range(origin_size[0] // output_size[0]):
            for j in range(origin_size[1] // output_size[1]):
                output_data = origin_data[i * output_size[0]:(i + 1) * output_size[0],
                              j * output_size[1]:(j + 1) * output_size[1], :]
                output_transform = (
                    x + j * output_x_step * output_size[1], output_x_step, 0, y + i * output_y_step * output_size[0], 0,
                    output_y_step)
                RASTER.write_image(
                    fn_out + '/dong_{}.tif'.format(10030 + i * (origin_size[1] // output_size[1]) + j + 1), output_data,
                    output_transform, proj)


def batch_clip(data: str, rois: list, output_path: str):
    """
    data->image path
    rois->a list of rois' path
    """
    data = gdal.Open(data)
    start_time = time.time()
    count = 0
    for roi in tqdm(rois):
        if count % 10 == 0:
            print('正在裁剪第{}幅影像'.format(count))
        name = roi.split('\\')[-1].split('.tif')[0]
        roi = gdal.Open(roi)
        if output_path is None:
            RASTER.clip_roi(data, roi, out_put_name=name)
        else:
            RASTER.clip_roi(data, roi, out_put_name=name, dstpath=output_path)
        count += 1
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('clipping time {}'.format(total_time_str))


def clip_sen12flood_ic():
    """裁剪flood的IC"""
    # 调用裁剪函数
    # raster_path = 'D:\\毕设实验数据\\flood_dataset\\sen12flood\\co_IC\\buhera_flood_IC.tif'  # 替换为您的影像路径
    # output_path = 'D:\\毕设实验数据\\flood_dataset\\sen12flood\\ic\\flood\\'  # 替换为您希望保存裁剪影像的路径

    # areas=['0'+str(i) for i in range(295,337)]+['46','4','53','32','6','51','56'] #australia area,desc
    # areas=['0'+str(i) for i in range(170,177)]+['22'] #naciaia iw3,asc
    # areas=['0'+str(i) for i in range(160,170)]+['54'] #naciaia iw1,asc
    # areas=['0'+str(i) for i in range(140,160)]+['11','0','67'] #beira,desc
    # areas=['0'+str(i) for i in range(186,223)]+['43','21','61','16','15','34'] #chipinge,asc
    # areas=['0177','0178','1','5','0181','0182','41','0184'] #
    # with open('D:\\下载\\chromedownload\\SEN12FLOOD\\SEN12FLOOD\\S1list.json','r') as f:
    #     json_data=f.read()
    #     data=json.loads(json_data)  

    # for area in areas:
    #     if data.get(area,False):
    #         flag=False
    #         count=1
    #         while not flag:
    #             flag=data[area][str(count)]['FLOODING'] and data[area][str(count)]['orbit']=='DESCENDING'
    #             filename=data[area][str(count)]['filename']
    #             count+=1
    #         print(filename)
    #         roi=gdal.Open(os.path.join('D:\\下载\\chromedownload\\SEN12FLOOD\\SEN12FLOOD',area,filename+'_corrected_VV.tif'))
    #         clipped=gdal.Open(raster_path)
    #         RASTER.clip_roi(clipped,roi,'Buhera_'+area,output_path)

    # """裁剪noflood的IC"""
    # raster_path = 'D:\\毕设实验数据\\flood_dataset\\sen12flood\\co_IC\\australia_noflood_IC.tif'  # 替换为您的影像路径
    # output_path = 'D:\\毕设实验数据\\flood_dataset\\sen12flood\\ic\\noflood\\'  # 替换为您希望保存裁剪影像的路径
    # name='Australia'
    # roi_names=[i for i in os.listdir('D:\\毕设实验数据\\flood_dataset\\sen12flood\\ic\\flood') if i.startswith(name) and i.endswith('.tif')]
    # for roi in roi_names:
    #     roi_data=gdal.Open(os.path.join('D:\\毕设实验数据\\flood_dataset\\sen12flood\\ic\\flood',roi))
    #     clipped=gdal.Open(raster_path)
    #     RASTER.clip_roi(clipped,roi_data,roi.split('.tif')[0],output_path)


def merge_overlap(area1, area2, dst_path, output_name):
    data1 = gdal.Open(area1)
    data2 = gdal.Open(area2)
    # # 获取ROI的几何信息（边界）
    roi_geotransform = data1.GetGeoTransform()
    width = data1.RasterXSize
    height = data1.RasterYSize
    proj = data1.GetProjection()
    bands = data1.RasterCount

    img_matrix = None
    for i in range(bands * 2):
        if i < 2:
            img = data1.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
        else:
            img = data2.GetRasterBand(i - 1).ReadAsArray(buf_type=gdal.GDT_Float32)
        img = np.expand_dims(img, 2)
        if i == 0:
            img_matrix = img
        else:
            img_matrix = np.concatenate((img_matrix, img), axis=2)
    output_data = np.full((height, width, bands), -1, dtype=np.float32)
    for band in range(bands):
        for i in range(height):
            for j in range(width):
                temp1 = img_matrix[i, j, band]
                temp2 = img_matrix[i, j, band + 2]
                if temp1 == -1:
                    output_data[i, j, band] = temp2
                else:
                    output_data[i, j, band] = temp1
    output_transform = roi_geotransform
    RASTER.write_image(dst_path + output_name + '.tif', output_data, output_transform, proj)

    # 清理，关闭数据集
    del data1, data2
    del output_data
    del img_matrix
    return


def modify_nanarea(path):
    data = gdal.Open(path)
    # # 获取ROI的几何信息（边界）
    roi_geotransform = data.GetGeoTransform()
    width = data.RasterXSize
    height = data.RasterYSize
    proj = data.GetProjection()
    bands = data.RasterCount
    img_matrix = None
    for i in range(bands):
        img = data.GetRasterBand(i + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
        img = np.expand_dims(img, 2)
        if i == 0:
            img_matrix = img
        else:
            img_matrix = np.concatenate((img_matrix, img), axis=2)
    output_data = np.full((height, width, bands), -1, dtype=np.float32)
    # 将NaN的区域赋值为整张图像的均值
    for i in range(bands):
        output_data[:, :, i] = img_matrix[:, :, i]
        nan_mask = np.isnan(img_matrix[:, :, i])
        mean_val = np.nanmean(img_matrix[:, :, i])
        output_data[:, :, i][nan_mask] = mean_val
    output_transform = roi_geotransform
    RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\sen12flood\\' + 'temp3\\' + path.split('\\')[-1], output_data,
                       output_transform, proj)

    # 清理，关闭数据集
    del data
    del output_data
    del img_matrix
    return


def create_valid_mask(path):
    data = gdal.Open(path)
    # # 获取ROI的几何信息（边界）
    roi_geotransform = data.GetGeoTransform()
    width = data.RasterXSize
    height = data.RasterYSize
    proj = data.GetProjection()

    img = data.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)
    output_data = np.full((height, width, 1), 1, dtype=np.float32)
    mask = np.where(np.isnan(img))
    output_data[mask] = 0
    output_transform = roi_geotransform
    RASTER.write_image('D:\\毕设实验数据\\flood_dataset\\sen12flood\\' + 'valid_mask\\' + path.split('\\')[-1],
                       output_data, output_transform, proj)

    # 清理，关闭数据集
    del data
    del output_data
    del img
    return


def create_image(path):
    width = 31102
    height = 32990
    xRes = 8.983152841195215e-05
    yRes = -8.983152841195215e-05
    minX = 112.14282951
    minY = 31.49825114
    step = 4096
    proj = 'GEOGCS["WGS 84",\
        DATUM["WGS_1984",\
            SPHEROID["WGS 84",6378137,298.257223563,\
                AUTHORITY["EPSG","7030"]],\
                    AUTHORITY["EPSG","6326"]],\
                        PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],\
                            UNIT["degree",0.0174532925199433,\
                                AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],\
                                    AUTHORITY["EPSG","4326"]]'
    xflag = yflag = 0
    for j in range(0, height, step):
        for i in range(0, width, step):
            minx = i * xRes + minX
            miny = j * yRes + minY
            maxx = minx + step * xRes
            maxy = miny + step * yRes
            geotransform = (minx, xRes, 0.0, miny, 0.0, yRes)
            output_data = np.full((step, step, 1), np.nan, dtype=np.float32)
            RASTER.write_image(
                'E:\\毕设实验数据\\flood_dataset\\study area\\rois\\' + str(yflag) + '_' + str(xflag) + '.tif',
                output_data, geotransform, proj)
            xflag += 1
        yflag += 1
        xflag = 0
    return


def main():
    name = '0619'
    data = "E:\\毕设实验数据\\flood_dataset\\study area\\hong\\predict_tif\\{}.tif".format(name)
    # roi_list=['E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\images\\roi\\'+i for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\images\\roi') if i.endswith('.tif')]
    roi_list = ['E:\\毕设实验数据\\flood_dataset\\study area\\hong\\long_time_water\\hong_1.tif']
    batch_clip(data, roi_list, 'E:\\毕设实验数据\\flood_dataset\\study area\\hong\\predict_tif\\')


if __name__ == '__main__':
    # main()
    # _4096dir=[i for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\chaohu_area_batch\\label\\') if i.endswith('.tif')]
    # _4096dir=['0718_sub1.tif']
    # for i in _4096dir:
    #     origin_data=gdal.Open(os.path.join('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\images\\',i))
    #     # prefix=i.split('.tif')[0]
    #     prefix='0718'
    #     fn_out='E:\\毕设实验数据\\flood_dataset\\study area\\dong\\images\\{}'.format(prefix)
    #     os.makedirs(fn_out,exist_ok=True)

    #     origin_transform=origin_data.GetGeoTransform()
    #     proj=origin_data.GetProjection()
    #     bands=origin_data.RasterCount
    #     # transform to np
    #     img_matrix = None
    #     for j in range(bands):
    #         data = origin_data.GetRasterBand(j + 1).ReadAsArray(buf_type=gdal.GDT_Float32)
    #         data = np.expand_dims(data, 2)
    #         if j == 0:
    #             img_matrix = data
    #         else:
    #             img_matrix = np.concatenate((img_matrix, data), axis=2)
    #     RASTER.split_image(fn_out,img_matrix,origin_transform,(2048,512),proj)#(row,col)

    # names=[os.path.join('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\0904',i)
    #        for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\0904\\') if i.endswith('.tif')]
    # gdal.Warp('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\0904.tif',names)

    # #结合SIA 对水体提取结果进行分析 flooded_area=extracted_water-(SIA_water+PIA_water)
    # sia_mask=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\sia\\sia_clipped.tif')
    # water=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\mask_0904_clipped.tif')
    # origin_transform=water.GetGeoTransform()
    # proj=water.GetProjection()
    # width=water.RasterXSize
    # height=water.RasterYSize
    # water_img=water.ReadAsArray(buf_type=gdal.GDT_Byte)
    # sia_img=sia_mask.ReadAsArray(buf_type=gdal.GDT_Byte)
    # output_data=np.full((height, width),0, dtype=np.byte)
    # NFA=(water_img==0)
    # FIA=(water_img==1) & (sia_img==1)
    # SIA=(water_img==1) & (sia_img==2)
    # PIA=(water_img==1) & (sia_img==3)
    # output_data[FIA]=3
    # output_data[SIA]=2
    # output_data[PIA]=1
    # output_data[NFA]=0
    # output_data=np.expand_dims(output_data, 2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_0904.tif',output_data,origin_transform,proj)

    # #统计淹没的次数情况，0-从没淹没，1-只淹没一次，2-淹没二到七次，3-一直淹没
    # _0718=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_0718.tif')
    # _0730=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_0730.tif')
    # _0811=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_0811.tif')
    # _0823=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_0823.tif')
    # _0904=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_0904.tif')
    # origin_transform=_0718.GetGeoTransform()
    # proj=_0718.GetProjection()

    # _0718_img=_0718.ReadAsArray(buf_type=gdal.GDT_Byte)
    # _0730_img=_0730.ReadAsArray(buf_type=gdal.GDT_Byte)
    # _0811_img=_0811.ReadAsArray(buf_type=gdal.GDT_Byte)
    # _0823_img=_0823.ReadAsArray(buf_type=gdal.GDT_Byte)
    # _0904_img=_0904.ReadAsArray(buf_type=gdal.GDT_Byte)

    # _0718_img[_0718_img!=3]=0
    # _0730_img[_0730_img!=3]=0
    # _0811_img[_0811_img!=3]=0
    # _0823_img[_0823_img!=3]=0
    # _0904_img[_0904_img!=3]=0

    # _0718_img[_0718_img==3]=1
    # _0730_img[_0730_img==3]=1
    # _0811_img[_0811_img==3]=1
    # _0823_img[_0823_img==3]=1
    # _0904_img[_0904_img==3]=1
    # # 统计淹没的次数情况，0-从没淹没，1-只淹没一次，2-淹没二到七次，3-一直淹没
    # output_data=np.full((_0718_img.shape[0], _0718_img.shape[1]),1, dtype=np.byte)
    # counts=_0718_img+_0730_img+_0811_img+_0823_img+_0904_img
    # output_data[counts==0]=0
    # output_data[counts>=5]=2

    # output_data=np.expand_dims(output_data, 2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\predict_tif\\flooded_area_change_state.tif',output_data,origin_transform,proj)

    ##分类dvdi 1-[0,inf), 2-[-2,0),3-[-5,-2),4-[-9,-5),5-[-13,-9),6(-inf,-13)
    # dvdi=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\hong\\dvdi\\hong_dvdi_month_7.tif')
    # dvdi_img_ndvi=dvdi.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)
    # dvdi_img_evi=dvdi.GetRasterBand(2).ReadAsArray(buf_type=gdal.GDT_Float32)
    # ndvi_mask1=dvdi_img_ndvi>=0
    # ndvi_mask2=(dvdi_img_ndvi>=-2) & (dvdi_img_ndvi<0)
    # ndvi_mask3=(dvdi_img_ndvi>=-5) & (dvdi_img_ndvi<-2)
    # ndvi_mask4=(dvdi_img_ndvi>=-9) & (dvdi_img_ndvi<-5)
    # ndvi_mask5=(dvdi_img_ndvi>=-13) & (dvdi_img_ndvi<-9)
    # ndvi_mask6=dvdi_img_ndvi<-13
    # evi_mask1=dvdi_img_evi>=0
    # evi_mask2=(dvdi_img_evi>=-2) & (dvdi_img_evi<0)
    # evi_mask3=(dvdi_img_evi>=-5) & (dvdi_img_evi<-2)
    # evi_mask4=(dvdi_img_evi>=-9) & (dvdi_img_evi<-5)
    # evi_mask5=(dvdi_img_evi>=-13) & (dvdi_img_evi<-9)
    # evi_mask6=dvdi_img_evi<-13

    # output_data_ndvi=np.full((dvdi_img_ndvi.shape[0], dvdi_img_ndvi.shape[1]),0, dtype=np.byte)
    # output_data_ndvi[ndvi_mask1]=1
    # output_data_ndvi[ndvi_mask2]=2
    # output_data_ndvi[ndvi_mask3]=3
    # output_data_ndvi[ndvi_mask4]=4
    # output_data_ndvi[ndvi_mask5]=5
    # output_data_ndvi[ndvi_mask6]=6

    # output_data_evi=np.full((dvdi_img_evi.shape[0], dvdi_img_evi.shape[1]),0, dtype=np.byte)
    # output_data_evi[evi_mask1]=1
    # output_data_evi[evi_mask2]=2
    # output_data_evi[evi_mask3]=3
    # output_data_evi[evi_mask4]=4
    # output_data_evi[evi_mask5]=5
    # output_data_evi[evi_mask6]=6

    # output_data_ndvi=np.expand_dims(output_data_ndvi, 2)
    # output_data_evi=np.expand_dims(output_data_evi, 2)
    # output_data=np.concatenate((output_data_ndvi,output_data_evi),axis=2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\hong\\dvdi\\dvdi_month_7.tif',output_data,dvdi.GetGeoTransform(),dvdi.GetProjection())

    # 掩膜掉SIA中的永久性水体区域
    # dvdi=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\dong_vrr_month_3.tif')
    # dvdi_img_ndvi=dvdi.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_img_evi=dvdi.GetRasterBand(2).ReadAsArray(buf_type=gdal.GDT_Byte)
    # sia_mask=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\sia\\sia_clipped.tif')
    # sia_mask_img=sia_mask.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # sia_mask1=sia_mask_img==3
    # dvdi_img_ndvi[sia_mask1]=0
    # dvdi_img_evi[sia_mask1]=0
    # output_data_ndvi=np.expand_dims(dvdi_img_ndvi, 2)
    # output_data_evi=np.expand_dims(dvdi_img_evi, 2)
    # output_data=np.concatenate((output_data_ndvi,output_data_evi),axis=2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_month_3_sia.tif',output_data,dvdi.GetGeoTransform(),dvdi.GetProjection())

    # 计算淹没次数区域和受灾严重区域的重叠度;淹没区域和受影响区域的重叠度
    # dvdi_7=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\dvdi\\dvdi_month_7.tif')
    # dvdi_8=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\dvdi\\dvdi_month_8.tif')
    # dvdi_9=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\dvdi\\dvdi_month_9.tif')
    # dvdi_10=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\dvdi\\dvdi_month_10.tif')
    # flood_state=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\wuh\\predict_tif\\flooded_area_change_state.tif')
    # flood_state_img=flood_state.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # # flood_state_mask=flood_state_img==2
    # dvdi_7_img=dvdi_7.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_8_img=dvdi_8.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_9_img=dvdi_9.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_10_img=dvdi_10.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)

    # # dvdi_state=np.zeros_like(flood_state_img)
    # # dvdi_state[dvdi_7_img>=4]=1
    # # dvdi_state[dvdi_8_img>=4]=1
    # # dvdi_state[dvdi_9_img>=4]=1
    # # dvdi_state[dvdi_10_img>=4]=1
    # # nums_dvdi=np.count_nonzero(dvdi_state==1)
    # # nums_flood=np.count_nonzero(flood_state_img==2)

    # # result=np.zeros_like(flood_state_img)
    # # result[((dvdi_7_img>=4) & (flood_state_mask))]=1
    # # result[((dvdi_8_img>=4) & (flood_state_mask))]=1
    # # result[((dvdi_9_img>=4) & (flood_state_mask))]=1
    # # result[((dvdi_10_img>=4) & (flood_state_mask))]=1
    # # nums=np.count_nonzero(result==1)
    # # print(nums,nums_dvdi,nums_flood)
    # # print(nums/(nums_dvdi+nums_flood))

    # flood_state_mask=((flood_state_img==2) | (flood_state_img==1))
    # dvdi_state=np.zeros_like(flood_state_img)
    # dvdi_state[dvdi_7_img==1]=1
    # dvdi_state[dvdi_8_img==1]=1
    # dvdi_state[dvdi_9_img==1]=1
    # dvdi_state[dvdi_10_img==1]=1
    # nums_dvdi=np.count_nonzero(dvdi_state==1)
    # nums_flood=np.count_nonzero(flood_state_mask==True)

    # result=np.zeros_like(flood_state_img)
    # result[((flood_state_mask) & (dvdi_8_img==1) & (dvdi_9_img==1) & (dvdi_10_img==1))]=1
    # # result[((dvdi_8_img==1) & (flood_state_mask))]=1
    # # result[((dvdi_9_img==1) & (flood_state_mask))]=1
    # # result[((dvdi_10_img==1) & (flood_state_mask))]=1
    # nums=np.count_nonzero(result==1)
    # print(nums,nums_dvdi,nums_flood)
    # print(nums/(nums_dvdi+nums_flood))

    # # 计算受灾类型面积和地物类型面积
    # dvdi_7=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\dvdi\\dvdi_month_7_sia.tif')
    # dvdi_8=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\dvdi\\dvdi_month_8_sia.tif')
    # dvdi_9=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\dvdi\\dvdi_month_9_sia.tif')
    # dvdi_10=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\dvdi\\dvdi_month_10_sia.tif')
    # lulc=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\lulc\\lulc.tif')
    # lulc_img=lulc.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)

    # dvdi_7_img=dvdi_7.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_8_img=dvdi_8.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_9_img=dvdi_9.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # dvdi_10_img=dvdi_10.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)

    # dvdi=[dvdi_7_img,dvdi_8_img,dvdi_9_img,dvdi_10_img]
    # for i in dvdi:
    #     re1=np.count_nonzero(((i==1) & ((lulc_img==4) | (lulc_img==3)))==True)
    #     re2=np.count_nonzero(((i==2) & ((lulc_img==4) | (lulc_img==3)))==True)
    #     re3=np.count_nonzero(((i==3) & ((lulc_img==4) | (lulc_img==3)))==True)
    #     re4=np.count_nonzero(((i==4) & ((lulc_img==4) | (lulc_img==3)))==True)
    #     re5=np.count_nonzero(((i==5) & ((lulc_img==4) | (lulc_img==3)))==True)
    #     re6=np.count_nonzero(((i==6) & ((lulc_img==4) | (lulc_img==3)))==True)
    #     with open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\dvdi\\rest_dvdi.txt','a') as f:
    #         f.write(str(re1)+','+str(re2)+','+str(re3)+','+str(re4)+','+str(re5)+','+str(re6)+'\n')
    #     print(re1,re2,re3,re4,re5,re6)

    #     print('-------------------')

    # 计算恢复情况，只要该像素达到恢复正常(较好、好、完全恢复)，则记为2，否则记为1，背景值为0
    # m11=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_month_11_sia.tif')
    # m12=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_month_12_sia.tif')
    # m1=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_month_1_sia.tif')
    # m2=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_month_2_sia.tif')
    # m3=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_month_3_sia.tif')
    # m11_img=m11.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # m12_img=m12.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # m1_img=m1.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # m2_img=m2.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # m3_img=m3.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # output_data=np.full((m11_img.shape[0], m11_img.shape[1]),0, dtype=np.byte)
    # time_data=np.full((m11_img.shape[0], m11_img.shape[1]),255, dtype=np.byte)

    # m11_mask_bad=((m11_img==1) | (m11_img==2) | (m11_img==3))
    # m11_mask_good=((m11_img==4) | (m11_img==5) | (m11_img==6))
    # m12_mask_bad=((m12_img==1) | (m12_img==2) | (m12_img==3))
    # m12_mask_good=((m12_img==4) | (m12_img==5) | (m12_img==6))
    # m1_mask_bad=((m1_img==1) | (m1_img==2) | (m1_img==3))
    # m1_mask_good=((m1_img==4) | (m1_img==5) | (m1_img==6))
    # m2_mask_bad=((m2_img==1) | (m2_img==2) | (m2_img==3))
    # m2_mask_good=((m2_img==4) | (m2_img==5) | (m2_img==6))
    # m3_mask_bad=((m3_img==1) | (m3_img==2) | (m3_img==3))
    # m3_mask_good=((m3_img==4) | (m3_img==5) | (m3_img==6))

    # output_data[m11_mask_bad]=1
    # output_data[m11_mask_good]=2
    # output_data[m12_mask_good]=2
    # output_data[m1_mask_good]=2
    # output_data[m2_mask_good]=2
    # output_data[m3_mask_good]=2

    # time_data[m3_mask_good]=5
    # time_data[m2_mask_good]=4
    # time_data[m1_mask_good]=3
    # time_data[m12_mask_good]=2
    # time_data[m11_mask_good]=1
    # time_data[output_data==1]=100

    # output_data=np.expand_dims(output_data, 2)
    # time_data=np.expand_dims(time_data, 2)
    # result=np.concatenate((output_data,time_data),axis=2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_recovery.tif',result,m11.GetGeoTransform(),m11.GetProjection())

    # #统计地物的恢复情况
    # data=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_recovery.tif')
    # data_img=data.GetRasterBand(2).ReadAsArray(buf_type=gdal.GDT_Byte)
    # lulc=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\lulc\\lulc.tif')
    # lulc_img=lulc.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Byte)
    # lulc_mask_5=lulc_img==5
    # lulc_mask_2=lulc_img==2
    # lulc_mask_6=lulc_img==6
    # lulc_mask_3_4=(lulc_img==3) | (lulc_img==4)

    # tree=np.full((data_img.shape[0], data_img.shape[1]),0, dtype=np.byte)
    # tree[lulc_mask_3_4]=data_img[lulc_mask_3_4]

    # tree=np.expand_dims(tree, 2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\rest.tif',tree,data.GetGeoTransform(),data.GetProjection())

    # names=[i for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\study area\\hong\\predict_tif\\0911') if i.endswith('.tif')]
    # path=[os.path.join('E:\\毕设实验数据\\flood_dataset\\study area\\hong\\predict_tif\\0911',i) for i in names]
    # gdal.Warp(R'E:\毕设实验数据\flood_dataset\study area\hong\predict_tif\merge_0911.tif',path)

    # vrr_recover=gdal.Open('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_recovery.tif')
    # vrr_recover_img=vrr_recover.GetRasterBand(2).ReadAsArray(buf_type=gdal.GDT_Byte)
    # mask100=vrr_recover==100
    # mask255=vrr_recover==255
    # result=vrr_recover_img
    # result[mask100]=6
    # result[mask255]=7
    # result=np.expand_dims(result, 2)
    # RASTER.write_image('E:\\毕设实验数据\\flood_dataset\\study area\\dong\\vrr\\vrr_recovery_100_255.tif',result,vrr_recover.GetGeoTransform(),vrr_recover.GetProjection())

    file1 = gdal.Open(R'E:\毕设实验数据\DEM\wuh_dem_clipped.tif')
    img1 = file1.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)
    result = 1 - (img1 - 6) / 381
    result[result < 0] = 0
    result[result > 1] = 1
    result = np.expand_dims(result, 2)
    RASTER.write_image(R'E:\毕设实验数据\DEM\wuh_dem_norm.tif', result, file1.GetGeoTransform(), file1.GetProjection())


    def recall(confusion_matrix):
        ntp = confusion_matrix[1, 1]
        nfn = confusion_matrix[1, 0]
        return ntp / (nfn + ntp)


    def precision(confusion_matrix):
        ntp = confusion_matrix[1, 1]
        nfp = confusion_matrix[0, 1]
        return ntp / (nfp + ntp)


    def F1(precision, recall):
        return 2 * (precision * recall) / (precision + recall)


    def accuracy(confusion_matrix):
        ntp = confusion_matrix[1, 1]
        ntn = confusion_matrix[0, 1]
        nfp = confusion_matrix[0, 1]
        nfn = confusion_matrix[1, 0]
        return (ntp + ntn) / (nfn + nfp + ntp + ntn)


    def iou(confusion_matrix):
        ntp = confusion_matrix[1, 1]
        nfp = confusion_matrix[0, 1]
        nfn = confusion_matrix[1, 0]
        return ntp / (nfn + nfp + ntp)

    # TN=FP=FN=TP=0
    # # image_names=[i for i in os.listdir('E:\\毕设实验数据\\flood_dataset\\sdwi\\label\\') if i.endswith('.tif')]
    # # names=['result.png','chaohu_10036.png','chaohu_10016.png','chaohu_10015.png','chaohu_10013.png','Bolivia_2982390.png']
    # image_names=['chaohu_10036.tif','chaohu_10016.tif','chaohu_10015.tif','chaohu_10013.tif']
    # for i in image_names:
    #     file1=gdal.Open('E:\\毕设实验数据\\flood_dataset\\chaohu_area_batch\\label\\'+i.split('.')[0]+'_label.tif')
    #     file2=gdal.Open('E:\\毕设实验数据\\flood_dataset\\ablation_intensity\\predict_imgs_tif\\'+i)
    #     label = file1.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)
    #     pred=file2.GetRasterBand(1).ReadAsArray(buf_type=gdal.GDT_Float32)

    #     # TN
    #     tn = ((pred == 0) & (label == 0)).sum()
    #     # FP
    #     fp = ((pred == 1) & (label == 0)).sum()
    #     # FN
    #     fn = ((pred == 0) & (label == 1)).sum()
    #     # TP
    #     tp = ((pred == 1) & (label == 1)).sum()
    #     TN+=tn
    #     FP+=fp
    #     FN+=fn
    #     TP+=tp
    # # print(TN,FP,FN,TP)
    # confusion_matrix = np.array([[TN, FP], [FN, TP]])
    # print(confusion_matrix)
    # print(recall(confusion_matrix))
    # print(precision(confusion_matrix))
    # print(F1(precision(confusion_matrix),recall(confusion_matrix)))
    # print(accuracy(confusion_matrix))
    # print(iou(confusion_matrix))
