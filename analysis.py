# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:28:53 2022

@author: ethan
"""

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf


TRUNC = (dt(2000, 1, 1), dt(2015, 1, 1))
COLS = {
    "gRGDP": ("gRGDPB2", "gRGDPB1", "gRGDPF0", "gRGDPF1", "gRGDPF2", "gRGDPF3"),
    "gPGDP": ("gPGDPB2", "gPGDPB1", "gPGDPF0", "gPGDPF1", "gPGDPF2", "gPGDPF3"),
    "UNEMP": ("UNEMPF0"),
}
ROMER_REP_DICT = {
    "diff_ffr": "DTARG",
    "ffr": "OLDTARG",
    "gRGDPB1": "GRAYM",
    "gRGDPF0": "GRAY0",
    "gRGDPF1": "GRAY1",
    "gRGDPF2": "GRAY2",
    "diff_gRGDPM": "IGRYM",
    "diff_gRGDP0": "IGRY0",
    "diff_gRGDP1": "IGRY1",
    "diff_gRGDP2": "IGRY2",
    "gPGDPB1": "GRADM",
    "gPGDPF0": "GRAD0",
    "gPGDPF1": "GRAD1",
    "gPGDPF2": "GRAD2",
    "diff_gPGDPM": "IGRDM",
    "diff_gPGDP0": "IGRD0",
    "diff_gPGDP1": "IGRD1",
    "diff_gPGDP2": "IGRD2",
    "UNEMPF0": "GRAU0",
    "shock": "RESID",
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
def get_intended_rates(drop_cols=True):
    if drop_cols:
        int_rate = pd.read_excel(
            "data/RomerandRomerDataAppendix.xls", usecols=[0, 1, 2]
        )
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
    mtg_date = int_rate.index.to_series()
    gb_date = df.index.to_series()

    # Corresponding MTG date associated with greenbook forecast
    int_rate["gb_date"] = pd.NaT

    for date in gb_date:
        i = mtg_date.searchsorted(date)
        int_rate.loc[mtg_date[i], "gb_date"] = date
        # print(i, "gb_date: {}, mtg_Date: {}".format(date, mtg_date.iloc[i]))
    int_rate = (
        int_rate.loc[~int_rate["gb_date"].isna(), :]
        .reset_index()
        .set_index(["gb_date"])
    )
    df["mtg_date"] = int_rate["fomc"]
    df["ffr_shock"] = int_rate["ff.shock.0"]
    df = (
        df.loc[~df["mtg_date"].isna(), :]
        .reset_index()
        .set_index(["mtg_date"])
    )
    
    print(df)
    return df


# We need to realign the quarters for the diff terms...
def calc_diffs(raw_data):
    stacked = (
        raw_data.drop(["ffr_shock", "UNEMPF0"], axis=1)
        .reset_index()
        .set_index(["mtg_date", "GBdate"])
        .stack()
        .reset_index()
    )

    stacked["var"] = stacked["level_2"].apply(lambda x: x[:-2])
    REL_QUARTER_DICT = {"B2": -2, "B1": -1, "F0": 0, "F1": 1, "F2": 2, "F3": 3}
    stacked["rel_quarter"] = stacked["level_2"].apply(
        lambda x: REL_QUARTER_DICT[x[-2:]]
    )

    get_abs_quarter = lambda x: (x.month - 1) // 3 + 1 + 4 * (x.year - TRUNC[0].year)
    stacked["abs_quarter_mtg"] = (
        stacked["mtg_date"].apply(get_abs_quarter) + stacked["rel_quarter"]
    )
    stacked["abs_quarter_GB"] = (
        stacked["GBdate"].apply(get_abs_quarter) + stacked["rel_quarter"]
    )

    groups = stacked.groupby("mtg_date")
    mtg_dates = raw_data.index.tolist()

    diffs = {}

    for m in range(1, len(mtg_dates)):
        # print(m, mtg_dates[m])
        group = groups.get_group(mtg_dates[m])
        last_group = groups.get_group(mtg_dates[m - 1])

        for i in range(-1, 3):
            temp = group.loc[group["rel_quarter"] == i, :].set_index("var")
            if not temp.any().any():
                continue
            j = temp["abs_quarter_GB"].iloc[0]
            temp_last = last_group.loc[last_group["abs_quarter_GB"] == j, :].set_index(
                "var"
            )

            diffs[(mtg_dates[m], i)] = round(temp[0] - temp_last[0], 4)
            calc = pd.DataFrame(
                {"diff": diffs[(mtg_dates[m], i)], "new": temp[0], "old": temp_last[0]}
            )
            # print(calc)

    diffs = pd.DataFrame(diffs).T.stack().reset_index()

    diffs["var"] = "diff_" + diffs["var"] + diffs["level_1"].astype(str)
    diffs["var"] = diffs["var"].apply(lambda x: x.replace("-1", "M"))
    diffs["mtg_date"] = diffs["level_0"]
    diffs = diffs.drop(["level_0", "level_1"], axis=1)

    diffs = pd.pivot_table(diffs, values=0, index="mtg_date", columns="var")

    return diffs


gb_data = get_var("gRGDP").join(get_var("gPGDP")).join(get_var("UNEMP"))
ffr_shock = pd.read_excel("data/NS.xlsx", sheet_name="shocks", index_col=0)
raw_data = align_dates(gb_data, ffr_shock)
diffs = calc_diffs(raw_data)
data = (
    raw_data.merge(diffs, how="left", on="mtg_date").drop(
        ["gRGDPB2", "gRGDPF3", "gPGDPB2", "gPGDPF3"], axis=1
    )
    # .dropna()
)
data = data.rename(ROMER_REP_DICT, axis=1)

model = smf.ols(
    (
        "DTARG ~ OLDTARG"
        "+ GRAYM + GRAY0 + GRAY1 + GRAY2"
        "+ IGRYM + IGRY0 + IGRY1 + IGRY2"
        "+ GRADM + GRAD0 + GRAD1 + GRAD2"
        "+ IGRDM + IGRD0 + IGRD1 + IGRD2"
        "+ GRAU0"
    ),
    data=data,
)

res = model.fit()
print(res.summary())
# data["shock"] = res.resid

# print(data)
#%%
# Here, we're working with FFR futures data

#%%
from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column


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


# Chart of a regression e.g. inflation vs money supply
def chart2(df):
    df = df.dropna()
    xdata, ydata, xrng, yrng = set_up(
        df.iloc[:, 0] / 100, df.iloc[:, 1] / 100, truncated=False, margins=0.005
    )

    p = figure(
        width=500,
        height=500,
        title="Reproduction Attempt: " + df.columns[0],
        x_axis_label="Romer and Romer " + df.columns[0],
        y_axis_label="Reproduced " + df.columns[0],
        y_range=yrng,
        x_range=xrng,
    )
    p.line(xrng, [0, 0], color="black")
    p.line([0, 0], yrng, color="black")

    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = "R = {:.4f}, Slope = {:.4f}".format(r_value, slope)
    p.line(xdata, xdata * slope + intercept, legend_label=leg, color="black")
    p.circle(xdata, ydata, color="blue", size=2)

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.legend.location = "bottom_right"

    export_png(p, filename="imgs/chart2.png")

    return p

