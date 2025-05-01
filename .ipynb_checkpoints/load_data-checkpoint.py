import pandas as pd
import numpy as np
import os, pickle
from scipy.io import loadmat

def load_data(args):
    home_root = "/home/melodia"
    EMBARC = f"{home_root}/data/EMBARC/EMBARC"
    FC_w0_root = f"{EMBARC}/Resting_fMRI_Connectivity/EMBARC_rsfMRI_fMRIPrep_Schaefer100_tp1_Baseline"
    
    cli_df = pd.read_csv(f'{home_root}/scPCAsCCA/data/cli_data.csv')
    
    subjs = []
    fcs = []
    for file in sorted(os.listdir(FC_w0_root)):
        if file.startswith('sub') and np.isin(file[4:10], cli_df['subj_ID']):
            file_path = f"{FC_w0_root}/{file}"
            fc_df = pd.read_csv(file_path, delimiter=' ', header=None)
            subjs.append(file[4:10])
            fcs.append(np.corrcoef(fc_df.T)[np.tril_indices(100, k=-1)])
    patient_EMBARC_df = pd.DataFrame({'subj_ID': subjs, 'w0_fc': fcs})

    if args.harm:
        hc_data = np.load('data/hc_data_harm.npy')
    else:
        hc_data = np.load('data/hc_data.npy')
    
    return hc_data, patient_EMBARC_df, cli_df