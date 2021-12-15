
# -*- coding: utf-8 -*-

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
    xdata, ydata, xrng, yrng = set_up(df['_'], df['_'], 
                                      truncated = False, margins = .005)
    
    p = figure(width = 500, height = 500,
               title="_" , 
               x_axis_label = '_', 
               y_axis_label = '_', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    p.line([0,0],yrng, color = 'black')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = 'R = {:.4f}, P-Value = {:.4e}, Slope = {:.4f}'.format(r_value,p_value,slope)
    p.line(xdata, xdata*slope + intercept, legend = leg, color = 'black')
    p.circle(xdata,ydata, color = 'blue',size = 2)
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter=NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")
    
    export_png(p, filename='images/chart2.png')
    
    return p
