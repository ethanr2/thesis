# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column


TRUNC = (dt(1969, 2, 1), dt(1996, 1, 1))
COLS = {
    "gRGDP": ("gRGDPB1", "gRGDPF0", "gRGDPF1", "gRGDPF2"),
    "gPGDP": ("gPGDPB1", "gPGDPF0", "gPGDPF1", "gPGDPF2"),
    "UNEMP": ("UNEMPF0"),
}

# Takes RGDP forecasts from the Greenbook dataset
def get_var(var):
    df = pd.read_excel("data/GBweb_Row_Format.xlsx", sheet_name=var)
    df["GBdate"] = pd.to_datetime(df["GBdate"], format="%Y%m%d")
    df = df.set_index("GBdate", drop=False)
    dates = df.pop("GBdate")

    df = df.loc[(TRUNC[0] < dates) & (dates < TRUNC[1]), COLS[var]]
    # df = df.loc[~df[var + "F2"].isnull(), :]
    return df


# Takes the intended FFR rates data directly from Romer and Romer's dataset
def get_intended_rates(drop_cols = True):
    if drop_cols:
        int_rate = pd.read_excel("data/RomerandRomerDataAppendix.xls", usecols=[0, 1, 2])
    else:
        int_rate = pd.read_excel("data/RomerandRomerDataAppendix.xls")
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
def align_dates(df, int_rate):
    ir_date = int_rate.pop("mtg_date")
    df_date = df.index.to_series()

    # Corresponding MTG date associated with greenbook forecast
    df["mtg_date"] = pd.NaT

    for date in ir_date:
        i = df_date.searchsorted(date) - 1
        # if i == 269:
        #     break
        df.loc[df_date[i], "mtg_date"] = date

        print("ir_date: {}, GB_Date: {}".format(date, df_date.iloc[i]))
    df = df.reset_index().set_index("mtg_date")
    df["ffr"] = int_rate["OLDTARG"]
    df["diff_ffr"] = int_rate["DTARG"]
    return df


raw_data = get_var("gRGDP").join(get_var("gPGDP")).join(get_var("UNEMP"))
int_rate = get_intended_rates()
raw_data = align_dates(raw_data, int_rate)

# For whatever reason there appears to be missing data in romer and romer's dataset for 1979-10-12? 
# just dropping it for now.
# Missing GB data on mtg_date 1971-08-24, GB_date 1971-8-20
raw_data = raw_data.dropna()
print(raw_data)
#%%
df = raw_data.copy()
for col in COLS["gRGDP"]:
    df["diff_" + col] = df[col].diff(1)
for col in COLS["gPGDP"]:
    df["diff_" + col] = df[col].diff(1)

df = df.dropna()

model = smf.ols(("diff_ffr ~ ffr + gRGDPB1 + gRGDPF0 + gRGDPF1 + gRGDPF2"
                 "+ diff_gRGDPB1 + diff_gRGDPF0 + diff_gRGDPF1 + diff_gRGDPF2" 
                 "+ gPGDPB1 + gPGDPF0 + gPGDPF1 + gPGDPF2"
                 "+ diff_gPGDPB1 + diff_gPGDPF0 + diff_gPGDPF1 + diff_gPGDPF2"
                 "+ UNEMPF0"),
                data=df
                )


res = model.fit()
res.summary()

#%%

# We need to realign the quarters for the diff terms...

romer_df = get_intended_rates(False)
temp = pd.concat([romer_df["IGRDM"], df["diff_gPGDPB1"]], axis = 1)
temp = pd.concat([romer_df["IGRDM"], df["diff_gPGDPB1"]], axis = 1)
temp
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
