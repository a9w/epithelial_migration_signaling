"""Calculate Spearmans correlation per sample for folders of sample csvs"""

# Import packages
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr

# Set data location, condition names (same as folder names)
DATA_DIR = "../data/colocalization/"
CONDITIONS = ["Ecad_Abi",
              "Fat2_Abi",
              "Lar_Abi",
              "Sema5c_Abi"]
OUT_DIR = DATA_DIR

# For each condition folder, measure the spearman's
# coefficient for each sample csv, output summary csv
for i, condition in enumerate(CONDITIONS):
    condition_dir = os.path.join(DATA_DIR + condition)
    condition_dir

    # Get names of all sample files in the folder
    file_names = sorted(os.listdir(condition_dir))
    file_names_csv = [name for name in file_names if '.csv' in name]
    basenames = []
    for file in file_names_csv:
        basenames.append(file.split('_reformatted.csv')[0])

    # Calculate spearmans coefficient for each sample
    spearmans = []
    for file in file_names_csv:
        data_path = os.path.join(condition_dir,file)
        intensities = pd.read_csv(data_path, index_col=0)
        ch1_intensity = intensities.iloc[:,1]
        ch2_intensity = intensities.iloc[:,2]
        spearmans.append(spearmanr(ch1_intensity, ch2_intensity, 
                                   nan_policy="omit")[0])

    # Make results df 
    col_names = ['basename', 'spearmans_r']
    spearmans_df = pd.DataFrame(list(zip(basenames, spearmans)),
                                columns = col_names)

    # Name and output the results df
    sample_num = basenames[0].split('_')[-1]
    condition = basenames[0].split('_' + sample_num)[0]
    out_path = OUT_DIR + 'Spearmans_r_' + condition + '.csv'

    out_path
    spearmans_df.to_csv(path_or_buf = out_path)    