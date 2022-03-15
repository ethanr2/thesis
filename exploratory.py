# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf




TRUNC = (dt(1969, 1, 1), dt(1997, 1, 1))
COLS = {
    "gRGDP": ("gRGDPB2", "gRGDPB1", "gRGDPF0", "gRGDPF1", "gRGDPF2", "gRGDPF3"),
    "gPGDP": ("gPGDPB2", "gPGDPB1", "gPGDPF0", "gPGDPF1", "gPGDPF2", "gPGDPF3"),
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
    ir_date = int_rate.pop("mtg_date")
    df_date = df.index.to_series()
    # return df_date
    # Corresponding MTG date associated with greenbook forecast
    df["mtg_date"] = pd.NaT

    for date in ir_date:
        i = df_date.searchsorted(date) - 1
        # if i == 269:
        #     break
        if df_date[i] == dt(year = 1979, month = 9, day = 12):
             #continue
             print(i, "ir_date: {}, GB_Date: {}".format(date, df_date.iloc[i]))
        df.loc[df_date[i], "mtg_date"] = date
        #print(i, "ir_date: {}, GB_Date: {}".format(date, df_date.iloc[i]))

    df = df.loc[~df["mtg_date"].isna(), :]
    df = df.reset_index().set_index(["mtg_date"])
    df["ffr"] = int_rate["OLDTARG"]
    df["diff_ffr"] = int_rate["DTARG"]
    return df


raw_data = get_var("gRGDP").join(get_var("gPGDP")).join(get_var("UNEMP"))

int_rate = get_intended_rates()
raw_data = align_dates(raw_data, int_rate)

# For whatever reason there appears to be missing data in romer and romer's dataset for 1979-10-12?
# just dropping it for now.
# Missing GB data on mtg_date 1971-08-24, GB_date 1971-8-20
#%%
# df = raw_data.copy()
# ir_date = int_rate["mtg_date"]
# df_date = df.index.to_series()

# # Corresponding MTG date associated with greenbook forecast
# int_rate["GBdate"] = pd.NaT

# for date in df_date:
#     i = ir_date.searchsorted(date)
#     # if i == 269:
#     #     break
#     #if df_date[i] == dt(year = 1979, month = 9, day = 12):
#          #continue
#     #     print(i, "ir_date: {}, GB_Date: {}".format(date, df_date.iloc[i]))
#     #df.loc[df_date[i], "mtg_date"] = date
#     print(i, "ir_date: {}, GB_Date: {}".format(ir_date[i], date))

# # df = df.loc[~df["mtg_date"].isna(), :]
# # df = df.reset_index().set_index(["mtg_date"])
# # df["ffr"] = int_rate["OLDTARG"]
# # df["diff_ffr"] = int_rate["DTARG"]


#%%

# We need to realign the quarters for the diff terms...
def calc_diffs(raw_data):
    stacked = (
        raw_data.drop(["ffr", "diff_ffr", "UNEMPF0"], axis=1)
        .reset_index()
        .set_index(["mtg_date", "GBdate"])
        .stack()
        .reset_index()
    )
    
    stacked["var"] = stacked["level_2"].apply(lambda x: x[:-2])
    REL_QUARTER_DICT = {"B2": -2, "B1": -1, "F0": 0, "F1": 1, "F2": 2, "F3": 3}
    stacked["rel_quarter"] = stacked["level_2"].apply(lambda x: REL_QUARTER_DICT[x[-2:]])
    
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
            temp_last = last_group.loc[last_group["abs_quarter_GB"] == j, :].set_index("var")
    
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
diffs = calc_diffs(raw_data)
data = (
    raw_data.merge(diffs, how="left", on="mtg_date")
    .drop(["gRGDPB2", "gRGDPF3", "gPGDPB2", "gPGDPF3"], axis=1)
    #.dropna()
)
print(data)
#%%
df = data.copy()


model = smf.ols(
    (
        "diff_ffr ~ ffr + gRGDPB1 + gRGDPF0 + gRGDPF1 + gRGDPF2"
        "+ diff_gRGDPM + diff_gRGDP0 + diff_gRGDP1 + diff_gRGDP2"
        "+ gPGDPB1 + gPGDPF0 + gPGDPF1 + gPGDPF2"
        "+ diff_gPGDPM + diff_gPGDP0 + diff_gPGDP1 + diff_gPGDP2"
        "+ UNEMPF0"
    ),
    data=df,
)

res = model.fit()
res.summary()



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
ROMER_REP_DICT = {
    "diff_ffr":"DTARG",
    "ffr":"OLDTARG",
    "gRGDPB1":"GRAYM",
    "gRGDPF0":"GRAY0", 
    "gRGDPF1":"GRAY1", 
    "gRGDPF2":"GRAY2",
    "diff_gRGDPM":"IGRYM",
    "diff_gRGDP0":"IGRY0",
    "diff_gRGDP1":"IGRY1", 
    "diff_gRGDP2":"IGRY2",
    "gPGDPB1":"GRADM",
    "gPGDPF0":"GRAD0",
    "gPGDPF1":"GRAD1",
    "gPGDPF2":"GRAD2",
    "diff_gPGDPM":"IGRDM", 
    "diff_gPGDP0":"IGRD0", 
    "diff_gPGDP1":"IGRD1", 
    "diff_gPGDP2":"IGRD2",
    "UNEMPF0":"GRAU0",
    "rep_shock":"RESID"
}
romer_data = get_intended_rates(False)
data["rep_shock"] = res.resid

#data= data.rename(ROMER_REP_DICT, axis = 1)

#data = data.drop("GBdate", axis = 1) == romer_data.drop("mtg_date", axis = 1)
#data = data.loc[romer_data.index,:]
#%%

# This draws plots comparing my dataset with romer and romers. 
plots = []


for key, value in ROMER_REP_DICT.items():
    temp = pd.concat([romer_data[value], data[key]], axis = 1).dropna()
    plots.append(chart2(temp))
    
    accuracy = sum(temp[key] == temp[value])/temp[key].size
    SSR = sum((temp[key] - temp[value])**2)
    
    print(key, value, temp[key].size, accuracy, SSR)

show(
     column(
         row(plots[0], plots[1], plots[2]),
         row(plots[3], plots[4], plots[5]),
         row(plots[6], plots[7], plots[8]),
         row(plots[9], plots[10], plots[11]),
         row(plots[12], plots[13], plots[14]),
         row(plots[15], plots[16], plots[17]),
         row(plots[18], plots[19])
         )
     )

# Kinda stuck. current plans:
# 1. Replicate the indicator from Romer and Romer's data directly.
# 2. Set up a method to determine percisely witch observations are different from RR
# 3. Investigate those observation to get a hint at whats going on.

#%%
# Step one, verify from Romer's own dataset. 

model = smf.ols(
    (
        "DTARG ~ OLDTARG"
        "+ GRAYM + GRAY0 + GRAY1 + GRAY2"
        "+ IGRYM + IGRY0 + IGRY1 + IGRY2"
        "+ GRADM + GRAD0 + GRAD1 + GRAD2"
        "+ IGRDM + IGRD0 + IGRD1 + IGRD2"
        "+ GRAU0"
    ),
    data=romer_data.dropna(),
)

res = model.fit()
res.summary()
temp = pd.concat([romer_data['RESID'], res.resid], axis = 1).dropna()
print(res.resid)
print(temp)
show(chart2(temp))

#%%
# Step two

