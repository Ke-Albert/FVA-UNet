import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 假设您的表格数据如下所示：
# Satellite, Date
# Satellite A, 2020-01-01
# Satellite B, 2020-01-02
# Satellite A, 2020-01-03
# ...

xls=pd.ExcelFile('D:\\桌面\\研究区数据情况.xlsx')
dan=pd.read_excel(xls,sheet_name='dan')
honghu=pd.read_excel(xls,sheet_name='honghu')

area=honghu
# 将日期列转换为datetime类型
area['Date'] = pd.to_datetime(area['Date'])

# 为不同的卫星类型分配颜色
satellite_colors = {
    'S1A': 'blue',
    'S2': 'red',
    'L8':'yellow',
    'L9':'pink',
    'GF1':'purple',
    'GF3':'brown'
}
start_date=datetime.datetime(2017,1,1)
end_date=datetime.datetime(2024,11,9)
dates = mdates.drange(start_date, end_date, datetime.timedelta(days=1))
# 绘制时间序列图
fig, ax = plt.subplots(figsize=(20, 6))
for satellite, color in satellite_colors.items():
    subset = area[area['Satellite'] == satellite]
    dates_subset = mdates.date2num(subset['Date'])
    ax.scatter(dates_subset, [satellite] * len(subset), color=color, label=satellite, alpha=0.6)

# 设置横轴坐标为日期格式，并显示为天数
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.set_xlim(mdates.date2num([start_date, end_date]))

# 设置y轴为卫星类型名称
ax.set_yticks(list(satellite_colors.keys()))
ax.set_yticklabels(list(satellite_colors.keys()))

# 设置图例
ax.legend()

# 设置标题和轴标签
ax.set_title('Satellite Data Visualization'+'--honghu')
ax.set_xlabel('Date')
ax.set_ylabel('Satellite')

# 显示图表
plt.show()