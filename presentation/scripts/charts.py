# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:05:58 2022

@author: ethan
"""

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from bokeh.io import export_png,output_file,show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet,ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column

df = (pd.read_csv("fredgraph.csv", 
                 na_values=".", 
                 index_col="DATE", 
                 parse_dates = True)
      .dropna()
      .iloc[:-8,:]
      /100
)
df

NIUred = (200,16,46)
NIUpantone = (165,167,168)
#%%
def set_up(x, y, truncated = True, margins = None):
    if truncated: 
        b = (3 * y.min() - y.max())/2
    else:
        b = y.min()
    if margins == None:    
        xrng = (x.min(),x.max())
        yrng = (b,y.max())
    else:
        xrng = (x.min() - margins,x.max() + margins)
        yrng = (b - margins,y.max() + margins)
        
    x = x.dropna()
    y = y.dropna()
    
    return(x,y,xrng,yrng)

# Chart of non-stationary time series, e.g. NGDP from 2008 to 2020    
def chart0(df):
    xdata, ydata, xrng, yrng = set_up(df.index,df['___'])
    
    p = figure(width = 1000, height = 500,
               title= '____', 
               x_axis_label = 'Date', x_axis_type = 'datetime',
               y_axis_label = '', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    
    p.line(xdata,ydata, color = 'blue', legend = '')
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = 'top_left'
    p.ygrid.grid_line_color = None
    p.yaxis.formatter=NumeralTickFormatter(format="____")
    
    export_png(p,filename ='imgs/chart0.png')

    return p

# Chart of approximately stionary time series, e.g. PCE-Core inflation from 2008 to 2020
def chart1(df):
    xdata, ydata, xrng, yrng = set_up(df.index, df['__'], truncated = False)
    
    p = figure(width = 1000, height = 500,
               title="_" , 
               x_axis_label = 'Date', x_axis_type = 'datetime',
               y_axis_label = '_', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    
    p.line(xdata,ydata, color = 'blue', legend = '_')
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = 'bottom_right'
    p.ygrid.grid_line_color = None
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")

    export_png(p, filename='imgs/chart1.png')

    return p

# Chart of a regression e.g. inflation vs money supply
def chart2(df):
    xdata, ydata, xrng, yrng = set_up(df['DFF'], df['GDPC1_PCA'], 
                                      truncated = False, margins = .005)
    xrng = (0, xrng[1])
    p = figure(width = 700, height = 500,
               title="Do Rate Hikes Decrease Real GDP? 1954 to 2019" , 
               x_axis_label = 'Federal Funds Effective Rate (Quarterly Average)', 
               y_axis_label = 'RGDP Growth (Annualized)', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black', width = 3)
    p.line([0,0],yrng, color = 'black', width = 3)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = 'R = {:.4f}, Slope = {:.4f}'.format(r_value, slope)
    p.line(xdata, xdata*slope + intercept, legend = leg, color = NIUpantone, width = 4)
    p.circle(xdata,ydata, color = NIUred, size = 5)
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter=NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.legend.label_text_font_size = "14pt"
    
    export_png(p, filename='imgs/ffrRgdp.png')
    
    return p

show(chart2(df))