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
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource, Label
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


def intro_charts(df, series, title, name, leg = None):
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
    if leg:
        p.line(xdata, ydata, color=NIUred, width=2, legend = leg)
    else:
        p.line(xdata, ydata, color=NIUred, width=2)

    p.xaxis[0].ticker.desired_num_ticks = 10
    # p.legend.location = "bottom_right"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.formatter = NumeralTickFormatter(format="0.00%")
    # p.xaxis.major_label_orientation = math.pi/4
    p.title.text_font_size = "16pt"
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    # p.legend.label_text_font_size = "14pt"
    if not leg:
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

chart1 = intro_charts(rr_df, "RESIDF", "Romer and Romer Indicator", "rr04.png")
chart2 = intro_charts(
    ns_df,
    "FFR_shock",
    "Gertler and Karadi Baseline Indicator",
    "gk15.png",  # Same data as GK15 but it came from NS18.
)
#%%
# Empirical Analysis
PRETTY_DICT = {
    "ffr_shock":r"\(FS_m\)",
    "GRAY0": r"\(\widetilde{\Delta y}_{m,0}\)",
    "GRAY1": r"\(\widetilde{\Delta y}_{m,1}\)",
    "GRAY2": r"\(\widetilde{\Delta y}_{m,2}\)",
    "IGRY0": r"\(\widetilde{\Delta y}_{m,0}-\widetilde{\Delta y}_{m-1,0}\)",
    "IGRY1": r"\(\widetilde{\Delta y}_{m,1}-\widetilde{\Delta y}_{m-1,1}\)",
    "IGRY2": r"\(\widetilde{\Delta y}_{m,2}-\widetilde{\Delta y}_{m-1,2}\)",
    "GRAD0": r"\(\tilde{\pi}_{m,0}\)",
    "GRAD1": r"\(\tilde{\pi}_{m,1}\)",
    "GRAD2": r"\(\tilde{\pi}_{m,2}\)",
    "IGRD0": r"\(\tilde{ \pi}_{m,0}-\tilde{ \pi}_{m-1,0}\)",
    "IGRD1": r"\(\tilde{ \pi}_{m,1}-\tilde{ \pi}_{m-1,1}\)",
    "IGRD2": r"\(\tilde{ \pi}_{m,2}-\tilde{ \pi}_{m-1,2}\)",
    "GRAU0": r"\(\tilde{u}_{m,0}\)",
}
full_sample = pd.read_pickle("data/processed_data/95sample_ns_data.pkl").iloc[:, 1:]
full_sample["ffr_shock"] = full_sample["ffr_shock"] * 10000
full_sample["pure_shock"] = full_sample["pure_shock"] * 100
coefs = pd.read_pickle("data/processed_data/bootstrap_ns_95sample_data.pkl")
model = smf.ols(
    (
        "ffr_shock ~ "
        "  GRAY0 + GRAY1 + GRAY2"
        "+ IGRY0 + IGRY1 + IGRY2"
        "+ GRAD0 + GRAD1 + GRAD2"
        "+ IGRD0 + IGRD1 + IGRD2"
        "+ GRAU0"
    ),
    data=full_sample,
)
print(model.fit().summary())
stage1_tab = Stargazer([model.fit()])
stage1_tab.rename_covariates(PRETTY_DICT)
stage1_tab = "\n".join(
    stage1_tab
    .render_latex()
    .replace(
        """{Dependent variable:}""",
        """{Dependent variable:} \(FS_m\)""",
    )
    .split("\n")[1:-1])
with open(TAB_PATH + "stage_1_tab.tex", "w") as file:
    file.write(stage1_tab)

#%%

full_sample["ffr_shock"] = full_sample["ffr_shock"] / 10000
full_sample["pure_shock"] = full_sample["pure_shock"] / 100
data = full_sample.loc[:, ["stock_returns", "pure_shock", "ffr_shock"]].dropna()
p2 = intro_charts(
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
    width=2,
    legend_label="Baseline FFR Futures Shock",
)
#show(p2)
export_png(p2, filename=CHART_PATH + "new_indicator.png")

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
stage2_1_mod = smf.ols("stock_returns ~ ffr_shock", data=df,)

stage2_2_mod = smf.ols("stock_returns ~ pure_shock", data=df,)
stage2_reg = Stargazer([stage2_1_mod.fit(), stage2_2_mod.fit()])
stage2_reg.rename_covariates(
    {"ffr_shock": """\(FS_m\)""", "pure_shock": """\(\hat{\epsilon}_m\)""",}
)
stage2_reg = "\n".join(
    stage2_reg.render_latex()
    .replace(
        """{Dependent variable:}""",
        """{Dependent variable:} \(\Delta \log{(\\text{S\&P500}_m)}\)""",
    )
    .split("\n")[1:-1]
)
with open(TAB_PATH + "stage_2_regression.tex", "w") as file:
    file.write(stage2_reg)
#%%

def make_hist(title, hist, edges, pdf, x):
    p = figure(
        title=title,
        width=1000,
        height=666,
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

diffs = coefs["pure_shock"] - coefs["ffr_shock"]
hist, edges = np.histogram(diffs, density=True, bins="auto")
nparam_density = stats.kde.gaussian_kde(diffs)
x = np.linspace(edges.min(), edges.max(), 1000)
nparam_density = nparam_density(x)
p4 = make_hist("Bootstrapped Distribution of δ - β", hist, edges, nparam_density, x)
p_val = sum(coefs["pure_shock"] - coefs["ffr_shock"] > 0) / len(coefs)
print(p_val)
p_label = Label(
    x=2, y=0.15, text="P(δ - β≥0)={:.1f}%".format(p_val * 100), text_font_size="14pt"
)
p4.add_layout(p_label)
export_png(p4, filename= CHART_PATH + "bootstrap_ns_95sample.png")
show(p4)
#%%
beta = stage2_1_mod.fit().params[1]
SE_beta = stage2_1_mod.fit().bse[1]

delta = stage2_2_mod.fit().params[1]
SE_delta = stage2_2_mod.fit().bse[1]

print(beta, delta)

H = (delta - beta) ** 2 / ( SE_delta ** 2 - SE_beta ** 2)
print(H)
print(1 - stats.chi2.cdf(H, 1))
