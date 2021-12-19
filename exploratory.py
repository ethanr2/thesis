# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column


TRUNC = (dt(1969, 2, 25), dt(1997, 1, 1))
COLS_RGDP = ("gRGDPB1", "gRGDPF0", "gRGDPF1", "gRGDPF2")
COLS_PGDP = ("gPGDPB1", "gPGDPF0", "gPGDPF1", "gPGDPF2")
# Takes RGDP forecasts from the Greenbook dataset
def get_rgdp():
    rgdp_df = pd.read_excel("data/GBweb_Row_Format.xlsx", sheet_name="gRGDP")
    rgdp_df["GBdate"] = pd.to_datetime(rgdp_df["GBdate"], format="%Y%m%d")
    rgdp_df = rgdp_df.set_index("GBdate", drop=False)
    dates = rgdp_df.pop("GBdate")

    rgdp_df = rgdp_df.loc[(TRUNC[0] < dates) & (dates < TRUNC[1]), COLS_RGDP]
    rgdp_df = rgdp_df.loc[~rgdp_df["gRGDPF2"].isnull(), :]
    return rgdp_df.sort_index()


# Takes the intended FFR rates data directly from Romer and Romer's dataset
def get_intended_rates():
    int_rate = pd.read_excel("data/RomerandRomerDataAppendix.xls", usecols=[0, 1, 2])
    int_rate["MTGDATE"] = int_rate["MTGDATE"].astype(str)

    # Weird parsing glitch because the dates were not zero padded. Might have cleaner fix.
    date_convert = lambda x: dt(
        year=int(x[-2:]) + 1900, day=int(x[-4:-2]), month=int(x[:-4])
    )

    int_rate["mtg_date"] = int_rate.pop("MTGDATE").apply(date_convert)
    dates = int_rate["mtg_date"]
    int_rate = int_rate.loc[(TRUNC[0] < dates) & (dates < TRUNC[1]), :]
    int_rate = int_rate.set_index("mtg_date", drop=False)
    return int_rate.sort_index()


# The GB forecasts aren't released at the same time as the meetings so this function makes sure
# that each row from the interest rate data is aligned with its proper row in the GB dataset
def align_dates(rgdp, int_rate):
    ir_date = int_rate.pop("mtg_date")
    rgdp_date = rgdp.index.to_series()

    # Corresponding MTG date associated with greenbook forecast
    rgdp["mtg_date"] = pd.NaT

    for date in ir_date:
        i = rgdp_date.searchsorted(date) - 1
        if i == 269:
            break
        rgdp.iloc[i, 4] = date

        print("ir_date: {}, rgdp_Date: {}".format(date, rgdp_date.iloc[i]))
    rgdp = rgdp.reset_index().set_index("mtg_date")
    rgdp["ffr"] = int_rate["OLDTARG"]

    return rgdp


rgdp = get_rgdp()
int_rate = get_intended_rates()
df = align_dates(rgdp, int_rate)
# for whatever reason there appears to be missing data in romer and romer's dataset for 1979-10-12? just dropping it for now.
df = df.dropna()
#%%
rgdp = get_rgdp()

df = pd.read_excel("data/GBweb_Row_Format.xlsx", sheet_name="gPGDP")
df["GBdate"] = pd.to_datetime(df["GBdate"], format="%Y%m%d")
df = df.set_index("GBdate", drop=False)
dates = df.pop("GBdate")

df = df.loc[(TRUNC[0] < dates) & (dates < TRUNC[1]), COLS_PGDP]
df = df.loc[~df["gPGDPF2"].isnull(), :]

rgdp = rgdp.join(df)
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


# Chart of non-stationary time series, e.g. NGDP from 2008 to 2020
def chart0(df):
    xdata, ydata, xrng, yrng = set_up(df.index, df["___"])

    p = figure(
        width=1000,
        height=500,
        title="____",
        x_axis_label="Date",
        x_axis_type="datetime",
        y_axis_label="",
        y_range=yrng,
        x_range=xrng,
    )
    p.line(xrng, [0, 0], color="black")

    p.line(xdata, ydata, color="blue", legend="")

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = "top_left"
    p.ygrid.grid_line_color = None
    p.yaxis.formatter = NumeralTickFormatter(format="____")

    export_png(p, filename="imgs/chart0.png")

    return p


# Chart of approximately stionary time series, e.g. PCE-Core inflation from 2008 to 2020
def chart1(df):
    xdata, ydata, xrng, yrng = set_up(df.index, df["__"], truncated=False)

    p = figure(
        width=1000,
        height=500,
        title="_",
        x_axis_label="Date",
        x_axis_type="datetime",
        y_axis_label="_",
        y_range=yrng,
        x_range=xrng,
    )
    p.line(xrng, [0, 0], color="black")

    p.line(xdata, ydata, color="blue", legend="_")

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = "bottom_right"
    p.ygrid.grid_line_color = None
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")

    export_png(p, filename="imgs/chart1.png")

    return p


# Chart of a regression e.g. inflation vs money supply
def chart2(df):
    xdata, ydata, xrng, yrng = set_up(df["_"], df["_"], truncated=False, margins=0.005)

    p = figure(
        width=500,
        height=500,
        title="_",
        x_axis_label="_",
        y_axis_label="_",
        y_range=yrng,
        x_range=xrng,
    )
    p.line(xrng, [0, 0], color="black")
    p.line([0, 0], yrng, color="black")

    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = "R = {:.4f}, P-Value = {:.4e}, Slope = {:.4f}".format(r_value, p_value, slope)
    p.line(xdata, xdata * slope + intercept, legend=leg, color="black")
    p.circle(xdata, ydata, color="blue", size=2)

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")

    export_png(p, filename="images/chart2.png")

    return p
