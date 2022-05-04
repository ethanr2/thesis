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
from scipy.io import loadmat

import statsmodels.api as sm
import statsmodels.formula.api as smf


TRUNC = (dt(1994, 2, 1), dt(2015, 1, 1))
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
STOCK_MKT_DATABASE = (
    "https://raw.githubusercontent.com/vijinho/sp500/master/csv/sp500.csv"
)

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
    df = df.loc[~df["mtg_date"].isna(), :].reset_index().set_index(["mtg_date"])

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


def get_data(source="data/NS.xlsx"):
    gb_data = get_var("gRGDP").join(get_var("gPGDP")).join(get_var("UNEMP"))
    # Reminder to switch to GK data later.
    ffr_shock = pd.read_excel(source, sheet_name="shocks", index_col=0)
    # ffr_shock = pd.read_csv(
    #     "data/gk_data/mps_updated.csv", index_col=0, parse_dates=True
    # )
    raw_data = align_dates(gb_data, ffr_shock)
    diffs = calc_diffs(raw_data)
    data = (
        raw_data.merge(diffs, how="left", on="mtg_date").drop(
            ["gRGDPB2", "gRGDPF3", "gPGDPB2", "gPGDPF3"], axis=1
        )
        # .dropna()
    )

    return data.rename(ROMER_REP_DICT, axis=1)


def construct_indicator(data, resample=False):
    data = data.copy()
    # if resample:
    #     data = data.sample(frac=resample)
    model = smf.ols(
        (
            "ffr_shock ~ "
            "  GRAYM + GRAY0 + GRAY1 + GRAY2"
            "+ IGRYM + IGRY0 + IGRY1 + IGRY2"
            "+ GRADM + GRAD0 + GRAD1 + GRAD2"
            "+ IGRDM + IGRD0 + IGRD1 + IGRD2"
            "+ GRAU0"
        ),
        data=data,
    )

    res = model.fit()
    if not resample:
        print(res.summary())

    data["pure_shock"] = res.resid / 100
    data["ffr_shock"] = data["ffr_shock"] / 100

    stock_df = pd.read_csv(STOCK_MKT_DATABASE, index_col="Date", parse_dates=True)

    data["stock_returns"] = np.log(stock_df.loc[data.index, "Close"]) - np.log(
        stock_df.loc[data.index, "Open"]
    )
    return data


# Bootstrap
def regressions(sample):
    mod1 = smf.ols("stock_returns ~ ffr_shock", data=sample).fit()
    mod2 = smf.ols("stock_returns ~ pure_shock", data=sample).fit()
    return pd.Series(
        [mod1.params["ffr_shock"], mod2.params["pure_shock"]],
        index=["ffr_shock", "pure_shock"],
    )


def bootstrap(N, frac):
    size = int(len(raw_data) * frac)
    sample_ind = [
        np.random.choice(raw_data.index, size=size, replace=False) for i in range(N)
    ]
    samples = []
    print("Preparing Samples.")
    t0 = time()
    for i in range(N):
        samples.append(construct_indicator(raw_data.loc[sample_ind[i], :], True))
        print(i, round(time() - t0, 2))

    samples = pd.Series(samples)
    coefs = samples.apply(regressions)
    return coefs


raw_data = get_data()
full_sample = construct_indicator(raw_data)
full_sample.to_pickle("data/processed_data/full_sample_ns_data.pkl")

# coefs = bootstrap(10000, 0.80)
# coefs.to_pickle("data/processed_data/bootstrap_ns_data.pkl")


#%%
# gk1 = pd.read_csv("data/gk_data/factor_data.csv",index_col=0)
# gk2 = loadmat("data/gk_data/DATASET.mat")
# ffr_shock = pd.read_excel("data/NS.xlsx", sheet_name="shocks", index_col=0)
# print(gk2)
#%%
