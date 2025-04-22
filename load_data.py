import pandas as pd
import numpy as np
import os, pickle
from scipy.io import loadmat

def load_data(args):
    home_root = "/home/melodia"
    EMBARC = f"{home_root}/data/EMBARC/EMBARC"
    FC_w0_root = f"{EMBARC}/Resting_fMRI_Connectivity/EMBARC_rsfMRI_fMRIPrep_Schaefer100_tp1_Baseline"
    
    cli_df = pd.read_csv(f'{home_root}/scPCAsCCA/data/cli_data_imputed.csv')
    
    subjs = []
    fcs = []
    for file in sorted(os.listdir(FC_w0_root)):
        if file.startswith('sub') and np.isin(file[4:10], cli_df['subj_ID']):
            file_path = f"{FC_w0_root}/{file}"
            fc_df = pd.read_csv(file_path, delimiter=' ', header=None)
            subjs.append(file[4:10])
            fcs.append(np.corrcoef(fc_df.T)[np.tril_indices(100, k=-1)])
    patient_EMBARC_df = pd.DataFrame({'subj_ID': subjs, 'w0_fc': fcs})
    # patient_EMBARC = np.stack(patient_EMBARC_df['w0_fc'])
    
    # EMBARC = f"{home_root}/data/EMBARC/EMBARC"
    # FC_w0_root = f"{EMBARC}/Resting_fMRI_Connectivity/EMBARC_rsfMRI_fMRIPrep_Schaefer100_tp1_Baseline"
    
    # TR_file = f"{EMBARC}/ClinicalData/embarc clinical variables summary 022617 gbc with outcome_gfupdated_3-5-18_excel95.xls"
    # demo_df = pd.read_excel(TR_file)
    
    # subjs = []
    # fcs = []
    # ages = []
    # for file in sorted(os.listdir(FC_w0_root)):
    #     if file.startswith('sub') and np.isin(file[4:10], demo_df[demo_df['severity1']==2]['subj_ID']):
    #         file_path = f"{FC_w0_root}/{file}"
    #         fc_df = pd.read_csv(file_path, delimiter=' ', header=None)
    #         subjs.append(file[4:10])
    #         fcs.append(np.corrcoef(fc_df.T)[np.tril_indices(100, k=-1)])
    # w0_df = pd.DataFrame({'subj_ID': subjs, 'w0_fc': fcs})
    # demo_df = pd.read_excel(TR_file)
    # demo_df = demo_df[demo_df['severity1']==2]
    # demo_df.rename(columns={'age_evaluation': 'age'}, inplace=True)
    # demo_df = demo_df[['subj_ID', 'age']]
    # EMBARC_df = w0_df.merge(demo_df, on='subj_ID')
    
    
    
    # AOMIC = f"{home_root}/data/AOMIC"
    # root = f"{AOMIC}/AOMIC-PIOP1/ROISignals/"
    # subjs = []
    # fcs = []
    # for subfolder in sorted(os.listdir(root)):
    #     for file in sorted(os.listdir(os.path.join(root, subfolder))):
    #         if 'Schaefer100' in file:
    #             fc_df = pd.read_csv(os.path.join(root, subfolder, file), delimiter=' ', header=None)
    #             subjs.append(subfolder[:8])
    #             fcs.append(np.corrcoef(fc_df.iloc[:,:100].T)[np.tril_indices(100, k=-1)])
    # w0_df = pd.DataFrame({'subj_ID': subjs, 'w0_fc': fcs})
    # hc_AOMIC = np.stack(w0_df['w0_fc'])
    # demo_df = pd.read_csv(f'{AOMIC}/participants_AOMIC-PIPO1.tsv', sep='\t')
    # demo_df.rename(columns={'participant_id': 'subj_ID'}, inplace=True)
    # demo_df = demo_df[['subj_ID', 'age']]
    # AOMIC_df = w0_df.merge(demo_df, on='subj_ID')
    
    
    
    # LEMON = f"{home_root}/data/LEMON"
    # data = loadmat(f"{LEMON}/fMRI_ROISignals.mat", squeeze_me=True)
    
    # fcs = []
    # for time_series in data['fMRI_ROISignals']:
    #     fcs.append(np.corrcoef(time_series.T)[np.tril_indices(100, k=-1)])
    # w0_df = pd.DataFrame({'subj_ID': data['subjectID_fMRI'], 'w0_fc': fcs})
    
    # demo_df = pd.read_csv(f'{LEMON}/DemographicData_LEMON.csv')
    # demo_df = demo_df[['Var1', 'Age']]
    # demo_df.rename(columns={'Var1': 'subj_ID', 'Age': 'age'}, inplace=True)
    # demo_df['age'] = demo_df['age'].map({'20-25': 22.5, '25-30': 27.5, '65-70': 67.5, '70-75': 72.5, '60-65': 62.5, '30-35': 32.5, '75-80': 77.5, '55-60':57.5, '35-40': 37.5})
    # LEMON_df = w0_df.merge(demo_df, on='subj_ID')
    
    
    # UCLA_CNP = f"{home_root}/data/UCLA_CNP"
    # root = f"{UCLA_CNP}/ROISignals/"
    
    # demo_df = pd.read_csv(f"{UCLA_CNP}/participants.tsv", sep='\t')
    # demo_df = demo_df[demo_df['diagnosis']=='CONTROL']
    
    # subjs = []
    # fcs = []
    # for subfolder in sorted(os.listdir(root)):
    #     for file in sorted(os.listdir(os.path.join(root, subfolder))):
    #         if 'Schaefer100' in file and np.isin(subfolder[:9], demo_df['participant_id']):
    #             fc_df = pd.read_csv(os.path.join(root, subfolder, file), delimiter=' ', header=None)
    #             subjs.append(subfolder[:9])
    #             fcs.append(np.corrcoef(fc_df.iloc[:,:100].T)[np.tril_indices(100, k=-1)])
    # w0_df = pd.DataFrame({'subj_ID': subjs, 'w0_fc': fcs})
    # demo_df = pd.read_csv(f"{UCLA_CNP}/participants.tsv", sep='\t')
    # demo_df = demo_df[demo_df['diagnosis']=='CONTROL']
    # demo_df.rename(columns={'participant_id': 'subj_ID'}, inplace=True)
    # demo_df = demo_df[['subj_ID', 'age']]
    # UCLA_CNP_df = w0_df.merge(demo_df, on='subj_ID')
    
    
    
    # with open('agematch.pkl', 'rb') as f:
    #     agelist = pickle.load(f)

    # hc_df = pd.concat([EMBARC_df, AOMIC_df, LEMON_df, UCLA_CNP_df], ignore_index=True).sort_values(by='age')
    # rows = []
    # for age in agelist:
    #     closest_row_index = (hc_df['age']-age).abs().idxmin()
    #     rows.append(hc_df.loc[closest_row_index])
    #     hc_df = hc_df.drop(index=closest_row_index).reset_index(drop=True)
    # hc_data = np.stack(pd.DataFrame(rows)['w0_fc'])

    if args.harm:
        hc_data = np.load('data/hc_data_harm.npy')
    else:
        hc_data = np.load('data/hc_data.npy')
    
    return hc_data, patient_EMBARC_df, cli_df