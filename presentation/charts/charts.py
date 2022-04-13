# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:03:50 2022

@author: ethan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:05:58 2022

@author: ethan
"""

from datetime import date
from datetime import datetime as dt
from time import time
import math

import numpy as np
import pandas as pd
from scipy import stats

from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column

df1 = (
    pd.read_excel(
        "data.xlsx", sheet_name="Sheet1", index_col="date_daily", parse_dates=True
    ).dropna()
    / 100
)

df2 = pd.read_excel("data.xlsx", sheet_name="Sheet2").dropna()
date_convert = lambda x: dt(
    year=int(x[-2:]) + 1900, day=int(x[-4:-2]), month=int(x[:-4])
)
df2.index = df2["MTGDATE"].astype(str).apply(date_convert)
df2 = df2 / 100

print(df2)

NIUred = (200, 16, 46)
NIUpantone = (165, 167, 168)
#%%
def set_up(x, y, truncated=True, margins=None):
    if truncated:
        b = (3 * y.min() - y.max()) / 2
    else:
        b = y.min()
    if margins == None:
        xrng = (x.min(), x.max())
        yrng = (b, y.max())
    else:
        xrng = (x.min() - margins, x.max() + margins)
        yrng = (b - margins, y.max() + margins)

    x = x.dropna()
    y = y.dropna()

    return (x, y, xrng, yrng)


# Chart of approximately stionary time series, e.g. PCE-Core inflation from 2008 to 2020
def chart1(df, series, title, name):
    xdata, ydata, xrng, yrng = set_up(df.index, df[series], truncated=False)
    scale = 1
    p = figure(
        width= int(1000 * scale),
        height=int(666 * scale),
        title=title,
        x_axis_label="Date",
        x_axis_type="datetime",
        y_range=yrng,
        x_range=xrng,
        toolbar_location=None,
    )
    p.line(xrng, [0, 0], color="black", width=1)

    p.line(xdata, ydata, color=NIUred, width=2)

    p.xaxis[0].ticker.desired_num_ticks = 10
    # p.legend.location = "bottom_right"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    #p.xaxis.major_label_orientation = math.pi/4
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    # p.legend.label_text_font_size = "14pt"
    export_png(p, filename=name)

    return p


# Chart of a regression e.g. inflation vs money supply
def chart2(df):
    xdata, ydata, xrng, yrng = set_up(
        df["DFF"], df["GDPC1_PCA"], truncated=False, margins=0.005
    )
    xrng = (0, xrng[1])
    p = figure(
        width=700,
        height=500,
        title="Do Rate Hikes Decrease Real GDP? 1954 to 2019",
        x_axis_label="Federal Funds Effective Rate (Quarterly Average)",
        y_axis_label="RGDP Growth (Annualized)",
        y_range=yrng,
        x_range=xrng,
    )
    p.line(xrng, [0, 0], color="black", width=3)
    p.line([0, 0], yrng, color="black", width=3)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = "R = {:.4f}, Slope = {:.4f}".format(r_value, slope)
    p.line(xdata, xdata * slope + intercept, legend=leg, color=NIUpantone, width=4)
    p.circle(xdata, ydata, color=NIUred, size=5)

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.legend.label_text_font_size = "14pt"

    export_png(p, filename="imgs/ffrRgdp.png")

    return p


show(
    column(
        chart1(df1, "FFR_shock", "Gertler and Karadi Baseline Indicator", "gk15.png"),
        chart1(df2, "RESIDF", "Romer and Romer Indicator", "rr04.png"),
    )
)
