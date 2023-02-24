"""
Functions for working with hemijunctions dataframes
(generated with measure_hemijunction_traits.py).

Each timelapse should be saved with name "condition_samplenumber.tif", and
related files should share the this name with added "keys." Hemijunctions data
for each timelapse should be saved as a csv with key "_data_hjs.csv".

The hemijunctions data should be imported with select_files function and stored
as a list of dictionaries, with each dictionary storing items related to one
sample. The _data_hjs.csv file for each sample should be imported and stored as
an entry with key "df_hjs".

"""

import numpy as np
import pandas as pd
from imageio import volread

def pool_df_all(datasets, df_key):
    """
    Concatenate all sample dfs entered in datasets into one df with cols
    added for basename, condition, sample_num.
    """
    basenames = []
    conditions = []
    sample_nums = []
    df_ls = []
    for dataset in datasets:
        basename = dataset["basename"]
        condition = dataset["condition"]
        sample_num = dataset["sample_num"]
        df = dataset[df_key]
        conditions.extend([condition] * len(df))
        sample_nums.extend([sample_num] * len(df))
        basenames.extend([basename] * len(df))
        df_ls.append(dataset[df_key])
    df_pooled = pd.concat(df_ls)
    df_pooled["basename"] = basenames
    df_pooled["condition"] = conditions
    df_pooled["sample_nums"] = sample_nums
    return df_pooled

def pool_df_by_condition(datasets, df_key, condition):
    """
    Concatenate all sample dfs of the same condition into one df with cols
    added for basename, condition, sample_num.
    """
    df_ls = []
    conditions = []
    basenames = []
    sample_nums = []
    for dataset in datasets:
        basename = dataset["basename"]
        dataset_condition =  dataset["condition"]
        if dataset_condition == condition:
            sample_num = dataset["sample_num"]
            df = dataset[df_key]
            df_ls.append(df)
            conditions.extend([condition] * len(df))
            sample_nums.extend([sample_num] * len(df))
            basenames.extend([basename] * len(df))
    df_pooled = pd.concat(df_ls)
    df_pooled["basename"] = basenames
    df_pooled["condition"] = conditions
    df_pooled["sample_nums"] = sample_nums
    return df_pooled

def add_prot_avg_len_col(df_hjs):
    """
    Calculate the average length of each hj (its area / its wiggly interface
    length) and add as new df column.
    """
    edge_len_nonstrt_um = df_hjs['edge_len_nonstrt_px'] * df_hjs.iloc[0]['um_per_px']
    edge_len_nonstrt_px = df_hjs['edge_len_nonstrt_px']
    hj_area_px2 = df_hjs['hj_area_px2']
    hj_area_um2 = hj_area_px2 * df_hjs.iloc[0]['um_per_px'] * df_hjs.iloc[0]['um_per_px']
    hj_avg_len_px = hj_area_px2 / edge_len_nonstrt_px
    hj_avg_len_um = hj_area_um2 / edge_len_nonstrt_um
    df_hjs["prot_avg_len_px"] = hj_avg_len_px
    df_hjs["prot_avg_len_um"] = hj_avg_len_um
    return df_hjs

def calc_prot_cutoff(df, statistic="prot_avg_len_um", percentile=98):
    """
    Calculate the value of the Xth percentile of a numeric df column.
    """
    cutoff = np.percentile(df[statistic], percentile)
    return cutoff

def make_prot_df(df_hjs, cutoff, statistic="prot_avg_len_um"):
    """
    From df_hjs, rows with a value above cutoff (int/float) in their statistic
    (str, header of a numeric column) will be included in df_prots.
    """
    df_prots = df_hjs[df_hjs[statistic]>cutoff]
    return df_prots

def make_and_store_prot_dfs(datasets,
                            df_key="df_hjs",
                            cutoff_condition="ck666",
                            statistic="prot_avg_len_um",
                            cutoff_percentile=98):
    df_cutoff_condition = pool_df_by_condition(datasets, df_key, cutoff_condition)
    cutoff = calc_prot_cutoff(df_cutoff_condition, statistic, cutoff_percentile)
    for dataset in datasets:
        df_prots = make_prot_df(dataset[df_key], cutoff, statistic)
        dataset["df_prots"] = df_prots
    return datasets

def get_coords_from_str(coord_str):
    """
    Turn coord tuples, which are turned into strings during df export to csv,
    back into tuples.
    """
    r = coord_str.split(" ")[0]
    r = r.split("(")[1]
    r = r.split(",")[0]
    r = int(r)

    c = coord_str.split(" ")[1]
    c = c.split(")")[0]
    c = int(c)
    return(r, c)

def store_hjs_in_dataset_dicts(datasets):
    for dataset in datasets:
        df_hjs = pd.read_csv(dataset["_data_hjs.csv"], index_col=0)
        df_hjs = add_prot_avg_len_col(df_hjs)
        dataset["df_hjs"] = df_hjs
    return datasets

def store_ims_in_dataset_dicts(datasets, file_key=".tif", dict_key="ims"):
    for dataset in datasets:
        ims = volread(dataset[file_key])
        dataset[dict_key] = ims
    return datasets
