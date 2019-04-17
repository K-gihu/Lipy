# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:32:16 2017

@author: lionking
"""

def set_ch():
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题