# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:20:21 2022

@author: ethan
"""

from datetime import date
from datetime import datetime as dt
from datetime import timedelta as td
import time

import numpy as np
import pandas as pd
from scipy import stats

from bokeh.io import export_png, output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column

URL = (
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
    "&symbol=VOO"
    "&datatype=csv"
    "&outputsize=full"
    "&apikey={}"
)
with open("apiKey.txt") as file:
    KEY = file.read().split(",")[1]



#%%
