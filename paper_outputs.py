# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 02:18:13 2022

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
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column

# Prepare results for paper
from stargazer.stargazer import Stargazer

TAB_PATH = "thesis_draft/latex_tables/"
CHART_PATH = "thesis_draft/charts/"

NIUred = (200, 16, 46)
NIUpantone = (165, 167, 168)
#%%
# Introduction
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


def intro_charts(df, series, title, name):
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
    p.line(xrng, [0, 0], color="black", width=1)

    p.line(xdata, ydata, color=NIUred, width=2)

    p.xaxis[0].ticker.desired_num_ticks = 10
    # p.legend.location = "bottom_right"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.formatter = NumeralTickFormatter(format="0.0%")
    # p.xaxis.major_label_orientation = math.pi/4
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    # p.legend.label_text_font_size = "14pt"
    export_png(p, filename=CHART_PATH + name)

    return p


ns_df = (
    pd.read_excel(
        "data/NS_95Sample.xlsx",
        sheet_name="PolicyNewsShocks1995",
        index_col="date_daily",
        parse_dates=True,
    ).dropna()
    / 100
)

rr_df = pd.read_excel("data/RomerandRomerDataAppendix.xls").dropna()
date_convert = lambda x: dt(
    year=int(x[-2:]) + 1900, day=int(x[-4:-2]), month=int(x[:-4])
)
rr_df.index = rr_df["MTGDATE"].astype(str).apply(date_convert)
rr_df = rr_df / 100
show(
    column(
        intro_charts(
            ns_df, "FFR_shock", "Nakamura and Steinsson Baseline Indicator", "ns15.png"
        ),
        intro_charts(rr_df, "RESIDF", "Romer and Romer Indicator", "rr04.png"),
    )
)
#%%
full_sample = pd.read_pickle("data/processed_data/full_sample_ns_data.pkl")
coefs = pd.read_pickle("data/processed_data/bootstrap.pkl")
model = smf.ols(
    (
        "ffr_shock ~ "
        "  GRAYM + GRAY0 + GRAY1 + GRAY2"
        "+ IGRYM + IGRY0 + IGRY1 + IGRY2"
        "+ GRADM + GRAD0 + GRAD1 + GRAD2"
        "+ IGRDM + IGRD0 + IGRD1 + IGRD2"
        "+ GRAU0"
    ),
    data=full_sample,
)
stage1_tab = Stargazer([model.fit()]).render_latex()
with open(TAB_PATH + "stage_1_tab.tex", "w") as file:
    file.write(stage1_tab)
#%%
# Summary Statistics for Stage 2
df = full_sample.loc[:, ["ffr_shock", "pure_shock", "stock_returns"]].dropna()
pretty_df = (
    df.rename(
        {
            "ffr_shock": """\(FS_m\)""",
            "pure_shock": """\(\hat{\epsilon}_m\)""",
            "stock_returns": """\(\Delta \log{(\text{S\&P500}_m)}\)""",
        },
        axis=1,
    )
    * 100
)
stage2_sum_stats = (
    pretty_df.describe()
    .to_latex(
        float_format="%.4f",
        # caption="""Summary statistics for our final dataset. Note that all variables are reported as percentages.""",
        # label="SumStats",
        escape=False,
    )
    .replace("%", "\%")
    .split("\n")
)
stage2_sum_stats[4] = stage2_sum_stats[4].replace(".0000", "")
stage2_sum_stats = "\n".join(stage2_sum_stats)

with open(TAB_PATH + "stage_2_summary_stats.tex", "w") as file:
    file.write(stage2_sum_stats)

#%%
stage3_1_mod = smf.ols("stock_returns ~ ffr_shock", data=df,)

stage3_2_mod = smf.ols("stock_returns ~ pure_shock", data=df,)
print(Stargazer([stage3_1_mod.fit(), stage3_2_mod.fit()]).render_latex())

H = (-7.154 + 6.518) ** 2 / (2.919 ** 2 - 2.601 ** 2)
1 - stats.chi2.cdf(H, 1)
