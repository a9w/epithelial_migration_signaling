"""
Import hemijunction data, calculate the protrusive/total hemijunction ratio
per egg chamber, output as a CSV. A protrusive hemijunction is one longer
than the 98th percentile of CK-666 hemijunction length.
"""

import numpy as np
import pandas as pd

from functions.utils import (select_files,
                            store_hjs_in_dataset_dicts,
                            add_prot_avg_len_col,
                            calc_prot_cutoff,
                            pool_df_by_condition,
                            make_prot_df)

# Set location of hemijunction data, settings for protrusivity cutoff calc
# Set location of hemijunction data, output
DATA_DIR = ("../data/membrane_protrusivity_polarity/sample_data/")
OUT_DIR = ("../data/membrane_protrusivity_polarity/")
CUTOFF_CONDITION = "ck666"
CUTOFF_PERCENTILE = 98

# Get paths to all hemijunction data in DATA_DIR
file_info = select_files(DATA_DIR, "_data_hjs.csv")

# Calculate the protrusion length cutoff
file_info = store_hjs_in_dataset_dicts(file_info)
df_cutoff_condition = pool_df_by_condition(file_info, "df_hjs", CUTOFF_CONDITION)
df_cutoff_condition = add_prot_avg_len_col(df_cutoff_condition)
prot_cutoff = calc_prot_cutoff(df_cutoff_condition, 
                                statistic="prot_avg_len_um", 
                                percentile=CUTOFF_PERCENTILE)

# Measure the protrusivity ratio of each sample
condition_ls = []
sample_num_ls = []
prot_hjs_ls = []
total_hjs_ls = []
prot_ratio_ls = []

for file in file_info:
    # Get dfs, sample info
    df_hjs = file["df_hjs"]
    sample_num = file["sample_num"]
    condition = file["condition"]

    # Calculate average protrusion length
    df_hjs = add_prot_avg_len_col(df_hjs)

    # Make df of protrusions 
    df_prots = make_prot_df(df_hjs, prot_cutoff, statistic="prot_avg_len_um")

    # Calculate the protrusivity ratio
    prot_hjs = len(df_prots)
    total_hjs = len(df_hjs)
    prot_ratio = prot_hjs/total_hjs

    # Add to lists
    condition_ls.append(condition)
    sample_num_ls.append(sample_num)
    prot_hjs_ls.append(prot_hjs)
    total_hjs_ls.append(total_hjs)
    prot_ratio_ls.append(prot_ratio)

# Construct df
col_names = ['condition', 'sample_num', 'prot_hjs', 'total_hjs', 'prot_ratio']
df_protrusivity = pd.DataFrame(list(zip(condition_ls, sample_num_ls, prot_hjs_ls,
                                        total_hjs_ls, prot_ratio_ls)),
                     columns = col_names)
df_protrusivity["cutoff_condition"] = CUTOFF_CONDITION
df_protrusivity["cutoff_percentile"] = CUTOFF_PERCENTILE
df_protrusivity["cutoff_prot_avg_len_um"] = prot_cutoff
df_sorted = df_protrusivity.sort_values(['condition', 'sample_num', "cutoff_condition",
                                        "cutoff_percentile", "cutoff_prot_avg_len_um"])
df_sorted.reset_index(inplace=True, drop=True)

# Output as CSV
df_sorted.to_csv(path_or_buf = OUT_DIR + "protrusivity_ratio_avg_len_sample.csv")
