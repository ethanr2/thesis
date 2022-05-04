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
