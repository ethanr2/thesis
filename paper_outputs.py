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

pretty_df = (
    df.rename(
        {
            "ffr_shock": """\(FS_m\)""",
            "pure_shock": """\(\hat{\epsilon}_m\)""",
            "stock_returns": "Stock Returns",
        },
        axis=1,
    )
    * 100
)
pretty_df.describe().to_latex(
    "2nd_stage_summary_stats.tex",
    float_format="%.4f",
    caption="""Summary statistics for our final dataset. Note that all variables are reported as percentages.""",
    label="SumStats",
    escape=False,
)

stage3_1_mod = smf.ols("stock_returns ~ ffr_shock", data=df,)

stage3_2_mod = smf.ols("stock_returns ~ pure_shock", data=df,)
print(Stargazer([stage3_1_mod.fit(), stage3_2_mod.fit()]).render_latex())

H = (-7.154 + 6.518) ** 2 / (2.919 ** 2 - 2.601 ** 2)
1 - stats.chi2.cdf(H, 1)