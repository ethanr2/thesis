# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 02:13:50 2022

@author: ethan
"""

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

import statsmodels.api as sm
import statsmodels.formula.api as smf

from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource, Label
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column

NIUred = (200, 16, 46)
NIUpantone = (165, 167, 168)

full_sample = pd.read_pickle("data/processed_data/full_sample.pkl")
coefs = pd.read_pickle("data/processed_data/bootstrap_ns_95sample_data.pkl")
coefs
#%%


#%%

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


def gdp_int_rate_chart(df):
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
        toolbar_location=None,
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

    export_png(p, filename="presentation/charts/ffrRgdp.png")

    return p


def base_line_pure_shock_timeseries(df, series, title, name, leg):
    xdata, ydata, xrng, yrng = set_up(df.index, df[series], truncated=False)
    scale = 1
    p = figure(
        width=int(1000 * scale),
        height=int(666 * scale),
        title=title,
        x_axis_label="Date",
        x_axis_type="datetime",
        y_range=yrng,
        x_range=xrng,
        toolbar_location=None,
    )
    p.line(xrng, [0, 0], color="black", width=3)

    p.line(
        df["pure_shock"].index,
        df["pure_shock"],
        color=NIUred,
        width=4,
        legend_label=leg,
    )

    # p.xaxis[0].ticker.desired_num_ticks = 10
    # p.legend.location = "bottom_right"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.formatter = NumeralTickFormatter(format="0.00%")
    # p.xaxis.major_label_orientation = math.pi/4
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.legend.label_text_font_size = "14pt"
    export_png(p, filename=name)

    return p


def reg_chart(data):
    xdata, ydata, xrng, yrng = set_up(
        data["ffr_shock"], data["stock_returns"], truncated=False
    )
    p = figure(
        width=700,
        height=500,
        title="Effect of Monetary Policy on the Stock Market",
        x_axis_label="Monetary Policy Shock",
        y_axis_label="Log Change in S&P500",
        y_range=yrng,
        x_range=xrng,
        toolbar_location=None,
    )
    p.line(xrng, [0, 0], color="black", width=3)
    p.line([0, 0], yrng, color="black", width=3)

    def add_reg_line(p, xdata, ydata, color, leg_title=""):
        slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
        leg = "R = {:.4f}, Slope = {:.4f}".format(r_value, slope)
        p.line(
            xdata,
            xdata * slope + intercept,
            legend_label=leg_title + leg,
            color=color,
            width=4,
        )
        p.circle(xdata, ydata, color=color, size=5)

    add_reg_line(p, xdata, ydata, NIUred, "Futures Shock: ")
    add_reg_line(p, data["pure_shock"], ydata, NIUpantone, "Purified Shock: ")

    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter = NumeralTickFormatter(format="0.00%")
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.legend.label_text_font_size = "14pt"

    export_png(p, filename="presentation/charts/reg_chart.png")
    return p


def make_hist(title, hist, edges, pdf, x):
    p = figure(
        title=title,
        width=700,
        height=500,
        toolbar_location=None,
        background_fill_color="white",
    )
    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color=NIUred,
        line_color=NIUred,
        alpha=1,
    )
    p.line(x, pdf, line_color=NIUpantone, line_width=6, alpha=0.90, legend_label="PDF")
    # p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")
    p.line([0, 0], [0, hist.max()], color="black", line_width=2)
    p.y_range.start = 0
    p.y_range.end = hist.max()
    p.x_range.start = x.min()
    p.x_range.end = x.max()
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = "δ-β"
    p.yaxis.axis_label = "Density"
    p.grid.grid_line_color = "white"
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.legend.label_text_font_size = "14pt"

    return p


df = (
    pd.read_csv("data/fredgraph.csv", na_values=".", index_col="DATE", parse_dates=True)
    .dropna()
    .iloc[:-8, :]
    / 100
)

p1 = gdp_int_rate_chart(df)


data = full_sample.loc[:, ["stock_returns", "pure_shock", "ffr_shock"]].dropna()
p2 = base_line_pure_shock_timeseries(
    data,
    "pure_shock",
    "Monetary Policy Shock Indicators",
    "new_indicator.png",
    leg="Purified Shock",
)
p2.line(
    data["ffr_shock"].index,
    data["ffr_shock"],
    color=NIUpantone,
    width=4,
    legend_label="Baseline FFR Futures Shock",
)
export_png(p2, filename="presentation/charts/pure_indicator.png")

p3 = reg_chart(data)

diffs = coefs["pure_shock"] - coefs["ffr_shock"]
hist, edges = np.histogram(diffs, density=True, bins="auto")
nparam_density = stats.kde.gaussian_kde(diffs)
x = np.linspace(edges.min(), edges.max(), 1000)
nparam_density = nparam_density(x)
p4 = make_hist("Bootstrapped Distribution of δ - β", hist, edges, nparam_density, x)
p_val = sum(coefs["pure_shock"] - coefs["ffr_shock"] > 0) / len(coefs)
print(p_val)
p_label = Label(
    x=1.0, y=0.15, text="P(δ - β≥0)={:.1f}%".format(p_val * 100), text_font_size="14pt"
)
p4.add_layout(p_label)
export_png(p4, filename="presentation/charts/bootstrap_ns_95sample.png")

show(column(p1, p2, p3, p4))

H = (-7.154 + 6.518) ** 2 / (2.919 ** 2 - 2.601 ** 2)
1 - stats.chi2.cdf(H, 1)
#%%

#%%
# df1 = (
#     pd.read_csv(
#         "data/GK_data/mps_updated.csv", index_col="Time", parse_dates=True
#     ).dropna()
#     / 100
# )

# df2 = pd.read_excel("data/RomerandRomerDataAppendix.xls").dropna()
# date_convert = lambda x: dt(
#     year=int(x[-2:]) + 1900, day=int(x[-4:-2]), month=int(x[:-4])
# )
# df2.index = df2["MTGDATE"].astype(str).apply(date_convert)
# df2 = df2["RESIDF"] / 100

# print(df2)

#%%


# Chart of a regression e.g. inflation vs money supply
def chart2(df):
    df = df.dropna()
    xdata, ydata, xrng, yrng = set_up(
        df.iloc[:, 0], df.iloc[:, 1], truncated=False, margins=0.005
    )

    p = figure(
        width=500,
        height=500,
        title="Effect of Monetary Policy on the Stock Market",
        x_axis_label="Monetary Policy Shock",
        y_axis_label="Log Change in S&P500 Price",
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


#%%

# Plot the effect of our indicators on the stock market
NIUred = (200, 16, 46)
NIUpantone = (165, 167, 168)
