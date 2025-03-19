import json
import copy
import os
import glob


# 将JSON数据转换为GeoJSON
def convert_to_geojson(json_data):
    # 创建一个空的FeatureCollection

    geojson_features = {'type':'FeatureCollection','name':'s1list','features':[]}

    # 遍历JSON数据中的每个事件
    for key, event in json_data.items():
        # 提取几何信息
        geometry = event.get('geo', {})
        properties={"ID":key,"count":event.get('count',0)}
        if_flood=False
        date=event['1']['date']
        for i in range(1,int(event.get('count',0))+1):
            if event[str(i)]['FLOODING']:
                if_flood=True
                date=event[str(i)]['date']
                break
        properties['FLOODING']=if_flood
        properties['date']=date
        if not geometry:
            continue
        # 创建GeoJSON Feature
        feature ={
            'type':'Feature',
            'properties':properties,
            'geometry':geometry,
        }
        # 将Feature添加到FeatureCollection中
        geojson_features['features'].append(copy.deepcopy(feature))

    return geojson_features

def read_json(path):
    with open(path,'r') as f:
        json_data=f.read()
        data=json.loads(json_data)
    return data

def compare(json_path,data_path):
    pattern=os.path.join(data_path,'*rural*.tif')
    #85.61899420856585_28.867885787429604
    areas=[i.split('rural_')[-1].strip('.tif') for i in glob.glob(pattern)]

    data=read_json(json_path)
    print(areas)
    for feature in data['features']:
        area_name=feature['properties']['location']
        with open(os.path.join(data_path,'statics.txt'),'a') as f:
            f.write(area_name+'\n')
        
        boundry=feature['geometry']['coordinates'][0]
        lonmin,lonmax,lanmin,lanmax=float('inf'),float('-inf'),float('inf'),float('-inf')
        for i in range(len(boundry)):
            lonmin,lonmax,lanmin,lanmax=min(boundry[i][0],lonmin),max(boundry[i][0],lonmax),\
                                        min(boundry[i][1],lanmin),max(boundry[i][1],lanmax)
            
        for area in areas:
            lon,lan=area.split('_')
            print(lon,lan+'\n')
            lon=float(lon)
            lan=float(lan)
            if check((lonmin,lonmax,lanmin,lanmax),lon,lan):
                with open(os.path.join(data_path,'statics.txt'),'a') as f:
                    f.write(area+'\n')

def check(four_corner,lon,lan):
    lonmin,lonmax,lanmin,lanmax=four_corner
    if lonmin<=lon<=lonmax and lanmin<=lan<=lanmax:
        return True
    return False

def one_extract(path):
    data=read_json(path)
    properties=data['properties']
    geometry=data['geometry']
    feature ={
            'type':'Feature',
            'properties':properties,
            'geometry':geometry,
        }
    # 将Feature添加到FeatureCollection中
    return feature

def main():
    # json_path="D:\\下载\\chromedownload\\SEN12FLOOD\\SEN12FLOOD\\S1list.json"
    # out_path="D:\\下载\\chromedownload\\SEN12FLOOD\\SEN12FLOOD\\S1list.geojson"
    # data=read_json(json_path)

    # # print(data['0063'])
    # # 转换并打印GeoJSON
    # geojson_result = convert_to_geojson(data)
    # result=json.dumps(geojson_result,indent=2)
    # with open(out_path,'w') as f:
    #     f.write(result)

    # compare("D:\\毕设实验数据\\flood_dataset\\sen1flood\\Sen1Floods11_Metadata.geojson",
    #         'D:\\毕设实验数据\\flood_dataset\\sen1flood\\JRC\\perm_water\\JRCPerm')
    
    geojson_features = {'type':'FeatureCollection','name':'s1flood11list','features':[]}
    path1='D:\\毕设实验数据\\flood_dataset\\sen1flood'
    path2='D:\\毕设实验数据\\flood_dataset\\sen1flood_4300+'

    endpath1='D:\\毕设实验数据\\flood_dataset\\sen1flood\\catalog\\catalog\\sen1floods11_hand_labeled_label'
    endpath2='D:\\毕设实验数据\\flood_dataset\\sen1flood\\catalog\\catalog\\sen1floods11_weak_labeled_label'
    #['Bolivia_103757_label', 'Bolivia_129334_label', 'Bolivia_195474_label', 'Bolivia_23014_label', 'Bolivia_233925_label']
    datas1=['_'.join(i.strip('.tif').split('_')[0:2])+'_label' for i in os.listdir(os.path.join(path1,'S1Hand_IC'))]
    datas2=['_'.join(i.strip('.tif').split('_')[0:2])+'_label' for i in os.listdir(os.path.join(path2,'S1_IC'))]
    for i in datas1:
        feature=one_extract(os.path.join(endpath1,i,i+'.json'))
        geojson_features['features'].append(copy.deepcopy(feature))
    for i in datas2:
        feature=one_extract(os.path.join(endpath2,i,i+'.json'))
        geojson_features['features'].append(copy.deepcopy(feature))
    with open('D:\\毕设实验数据\\flood_dataset\\sen1floods11.geojson','a') as f:
        result=json.dumps(geojson_features,indent=2)
        f.write(result)
    return
if __name__=='__main__':
    main()
