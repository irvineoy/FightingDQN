
# -*- coding:utf-8 -*-  
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import csv
# myfont = FontProperties(fname='/Library/Fonts/ipag.ttf')  

# font_prop = FontProperties(fname='/Library/Fonts/ipag.ttf')
# pl.rcParams['font.family'] = font_prop.get_name()
# , fontproperties=myfont

pl.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
pl.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
pl.rcParams['font.sans-serif']=['AppleGothic'] #用来正常显示中文标签
# pl.rcParams['font.family'] = ['Toppan Bunkyu Gothic'] #全体のフォントを設定
pl.rcParams["figure.figsize"] = [7, 4]
pl.rcParams['font.size'] = 10 #フォントサイズを設定 default : 12
pl.rcParams['xtick.labelsize'] = 15 # 横軸のフォントサイズ
pl.rcParams['ytick.labelsize'] = 15
pl.rcParams['axes.unicode_minus']=False

csvList = []
with open("./saved_networks/resultData.csv", 'r') as f:
	csvReader = csv.reader(f)
	for row in csvReader:
		if row[1] == 'R':
			continue
		csvList.append(int(row[1]))

countNumber = [i for i in range(len(csvList))]

# pl.scatter(countNumber, csvList, marker='o', color='k', label=u'Output voltage')
pl.plot(countNumber, csvList, label=u'Theoretical Output Voltage')
pl.legend(loc = 'upper left')
ax = pl.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
pl.show()