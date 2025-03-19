from osgeo import gdal

#reading TIFF image
class RASTER:

    @staticmethod
    def open(path):
        dataset=gdal.Open(path)
        return dataset

    def clip_roi(data,roi):
        # # 获取ROI的几何信息（边界）
        roi_geotransform = roi.GetGeoTransform()
        width=roi.RasterXSize
        height=roi.RasterYSize
        minX=roi_geotransform[0]
        minY=roi_geotransform[3]
        xRes=roi_geotransform[1]
        yRes=roi_geotransform[5]
        maxX=minX+width*xRes
        maxY=minY+height*yRes
        print(roi_geotransform)

        data_geotransform=data.GetGeoTransform()
        data_width=data.RasterXSize
        data_height=data.RasterYSize
        data_minx=data_geotransform[0]
        data_miny=data_geotransform[3]
        data_maxx=data_minx+data_width*xRes
        data_maxy=data_miny+data_height*yRes
        bands=data.RasterCount
        print(data_geotransform)

        # 定义输出数据集的参数
        # output_dataset = gdal.Warp('clipped_output.tif', data,
        #                             outputBounds=[minX,minY,maxX,maxY],
        #                             width=width,height=height,
        #                             yRes=yRes,
        #                             dstNodata=-999
        #                             )

        # # 保存裁剪后的图像到磁盘上
        # output_dataset_filename = 'clipped_output.tif'
        # output_dataset = output_dataset.GetDriver().CreateCopy(output_dataset_filename, output_dataset)
        # output_dataset.FlushCache()  # 确保数据写入磁盘

        # 清理，关闭数据集
        del data
        del roi
        # del output_dataset

def main():
    roi=RASTER.open('D:\毕设实验数据\\flood_dataset\sen1flood\S1Hand\\Ghana_161233_S1Hand.tif')
    data=RASTER.open("D:\毕设实验数据\\flood_dataset\sen1flood\co_IC\\Ghana_161233_S1Hand.tif")
    RASTER.clip_roi(data,roi)

if __name__=='__main__':
    main()