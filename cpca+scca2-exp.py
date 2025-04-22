from contrastive import CPCA
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from itertools import product
from load_data import load_data
import scipy.stats
import pickle
import argparse

from numpy.linalg import svd
from sklearn.decomposition import PCA

from sparsecca.sparsecca import cca_ipls
from sparsecca.sparsecca import cca_pmd
from sparsecca.sparsecca import multicca_pmd
from sparsecca.sparsecca import pmd

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--harm', action='store_true')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--penaltyu', type=float, default=0.6)
    parser.add_argument('--penaltyv', type=float, default=0.4)

    parser.add_argument('--seed', type=int, default=2014)
    args = parser.parse_args()

    return args

args = get_args()

def get_v_top(mdl, alpha):
    n_components = mdl.n_components
    sigma = mdl.fg_cov - alpha*mdl.bg_cov
    w, v = np.linalg.eig(sigma)
    eig_idx = np.argpartition(w, -n_components)[-n_components:]
    eig_idx = eig_idx[np.argsort(-w[eig_idx])]
    # from IPython import embed; embed(); exit()
    v_top = v[:,eig_idx].astype(np.float32)

    try:
        weight = mdl.pca.components_
        v_top = weight.T@v_top
    except:
        pass
    
    return v_top

hc_data, patient_EMBARC_df, cli_df = load_data(args)
patient_EMBARC_df = patient_EMBARC_df.groupby('subj_ID').apply(lambda l: np.sum(l)/len(l)).reset_index()
# selected_features = ['hamd_36', 'hamd_score_24', 'asrm_score2', 'aaq_score_result', 'ctqscore_ea', 'ctqscore_en', 'ctqscore_pa', 'ctqscore_pn', \
#                 'ctqscore_sa', 'ctqscore_val', 'cast_score_total', 'chrtp_propensity_score', 'chrtp_risk_score', \
#                 'masq2_score_aa', 'masq2_score_ad', 'masq2_score_gd', 'MASQ_AD_regressout_GD_AA', 'mdqscore_total', \
#                 'neo2_score_ag', 'neo2_score_co', 'neo2_score_ex', 'neo2_score_ne', 'neo2_score_op', 'qids_eeg_score', \
#                 'scq_total_score', 'shaps_total_continuous', 'shaps_total_dichotomous', \
#                 'sas_overall_mean', 'stai_eeg_final_score', 'stai_post_final_score', 'stai_pre_final_score', 'sapas_score', \
#                 'vas1', 'vams_hap1', 'vams_wit1', 'vams_rel1', 'vams_soc1']
# selected_features = ['hamd_36', 'asrm_score2', 'aaq_score_result', 'ctqscore_ea', 'ctqscore_en', 'ctqscore_pa', 'ctqscore_pn', \
#                 'ctqscore_sa', 'cast_score_total', 'chrtp_propensity_score', 'chrtp_risk_score', \
#                 'masq2_score_aa', 'masq2_score_ad', 'masq2_score_gd', 'mdqscore_total', \
#                 'neo2_score_ag', 'neo2_score_co', 'neo2_score_ex', 'neo2_score_ne', 'neo2_score_op', 'qids_eeg_score', \
#                 'scq_total_score', 'shaps_total_continuous', \
#                 'sas_overall_mean', 'stai_pre_final_score', 'sapas_score']
selected_features = ['hamd_36', 'asrm_score2', 'aaq_score_result', 'ctqscore_ea', 'ctqscore_en', 'ctqscore_pa', 'ctqscore_pn', \
                'ctqscore_sa', 'cast_score_total', 'chrtp_propensity_score', 'chrtp_risk_score', \
                'masq2_score_aa', 'masq2_score_ad', 'masq2_score_gd', 'mdqscore_total', \
                'neo2_score_ag', 'neo2_score_co', 'neo2_score_ex', 'neo2_score_ne', 'neo2_score_op', 'qids_eeg_score', \
                'scq_total_score', 'shaps_total_continuous', \
                'sas_overall_mean', 'sapas_score']
N_scales = len(selected_features)
full_patient_df = patient_EMBARC_df.merge(cli_df, on='subj_ID')
patient_fc = np.stack(full_patient_df['w0_fc'])

foreground = patient_fc
background = hc_data
cli_data = full_patient_df[selected_features].to_numpy()

params = product([args.seed], [args.alpha], [args.penaltyu], [args.penaltyv])

for seed, alpha, penaltyu, penaltyv in params:
    print(f'Current params: seed={seed}, alpha={alpha}, penaltyu={penaltyu}, penaltyv={penaltyv}')
    np.random.seed(seed)

    # t = time.time()
    mdl = CPCA(n_components=200)
    fg_mean = foreground.mean(axis=0)
    fg_std = foreground.std(axis=0)
    mdl.fit(foreground, background, preprocess_with_pca_dim=99999)

    v_top_all = get_v_top(mdl, alpha)
    cpcaed = ((foreground-fg_mean)/fg_std).dot(v_top_all)
    # cpcaed = mdl.cpca_alpha((foreground-fg_mean)/fg_std, alpha=alpha).astype(np.float32)
    
    cli_mean = cli_data.mean(axis=0)
    cli_std = cli_data.std(axis=0)
    cli_data_normed = (cli_data - cli_mean) / cli_std
    
    X = cpcaed
    Z = cli_data_normed

    pca = PCA(n_components=0.8)
    Z = pca.fit_transform(Z)
    N_scales = Z.shape[1]
    
    U, V, D = pmd(X.T @ Z, K=N_scales, penaltyu=penaltyu, penaltyv=penaltyv, standardize=False)
    
    Rs_all = []
    Us_all = []
    Vs_all = []
    
    for component_i in range(N_scales):
        x_weights = U[:, component_i]
        z_weights = V[:, component_i]
        corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
        Rs_all.append(corrcoef)
        Us_all.append(x_weights/np.max(np.abs(x_weights)))
        Vs_all.append(z_weights/np.max(np.abs(z_weights)))

    Vs_all = np.array(Vs_all)@pca.components_
    
    runs = 10
    reses = []
    idx_tests = []
    for run_i in range(runs):
        subj_ID = full_patient_df['subj_ID']
        subj_uniq = np.unique(subj_ID)
        n_subj = subj_uniq.shape[0]
        subj_shuffled = subj_uniq[np.random.permutation(n_subj)]
        fold = 10
        
        Rss, Uss, Vss, test_Uss, test_Vss = [], [], [], [], []
        v_tops = []
        for fold_i in tqdm(range(fold)):
            subj_test = subj_shuffled[int(n_subj*fold_i/fold):int(n_subj*(fold_i+1)/fold)]
            idx_train = ~np.isin(subj_ID, subj_test)
            idx_test = np.isin(subj_ID, subj_test)
            idx_tests.append(idx_test)
            foreground_train = foreground[idx_train]
            foreground_test = foreground[idx_test]
            
            fg_mean = foreground_train.mean(axis=0)
            fg_std = foreground_train.std(axis=0)
            
            mdl = CPCA(n_components=200)
            mdl.fit(foreground_train, background, preprocess_with_pca_dim=99999)
            v_top = get_v_top(mdl, alpha)
            v_tops.append(v_top)
            cpcaed_train = ((foreground_train-fg_mean)/fg_std).dot(v_top)
            cpcaed_test = ((foreground_test-fg_mean)/fg_std).dot(v_top)
            # cpcaed_train = mdl.cpca_alpha((foreground_train-fg_mean)/fg_std, alpha=alpha).astype(np.float32)
            # cpcaed_test = mdl.cpca_alpha((foreground_test-fg_mean)/fg_std, alpha=alpha).astype(np.float32)
        
            cli_train = cli_data[idx_train]
            cli_test = cli_data[idx_test]
            
            cli_mean = cli_train.mean(axis=0)
            cli_std = cli_train.std(axis=0)
            cli_train_normed = (cli_train - cli_mean) / cli_std
            cli_test_normed = (cli_test - cli_mean) / cli_std

            pca = PCA(n_components=0.8)
            cli_train_normed = pca.fit_transform(cli_train_normed)
            N_scales = cli_train_normed.shape[1]
            # from IPython import embed; embed(); exit()
            cli_test_normed = cli_test_normed@pca.components_.T
            
            X = cpcaed_train
            Z = cli_train_normed
            
            U, V, D = pmd(X.T @ Z, K=N_scales, penaltyu=penaltyu, penaltyv=penaltyv, standardize=False)
        
            Rs = []
            Us = []
            Vs = []
            test_Us = []
            test_Vs = []
            
            for component_i in range(N_scales):
                x_weights = U[:, component_i]
                z_weights = V[:, component_i]
                corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Z.T))[0, 1]
                Rs.append(corrcoef)
                Us.append(x_weights/np.max(np.abs(x_weights)))
                Vs.append(z_weights/np.max(np.abs(z_weights)))
                # combined_x_weights = np.dot(v_top, x_weights)
                # test_Us.append(np.dot(((foreground_test-fg_mean)/fg_std), combined_x_weights/np.sum(np.abs(combined_x_weights))))
                # test_Vs.append(np.dot(cli_test_normed, z_weights/np.max(np.abs(z_weights))))
                test_Us.append(np.dot(x_weights, cpcaed_test.T))
                test_Vs.append(np.dot(z_weights, cli_test_normed.T))
        
            Rss.append(Rs)
            Uss.append(Us)
            Vss.append(np.array(Vs)@pca.components_)
            test_Uss.append(test_Us)
            test_Vss.append(test_Vs)
            
        reses.append((np.array(Rss), np.array(Uss), np.array(Vss), test_Uss, test_Vss, v_tops))
    folder = {True: 'harm', False: 'vams'}[args.harm]
    # with open(f'results/{folder}/alpha_{alpha}_penaltyu_{penaltyu}_penaltyv_{penaltyv}_seed_{seed}@B32.pkl', 'wb') as f:
    # with open(f'results/alpha0.8.pkl', 'wb') as f:
    #     pickle.dump((reses, np.array(Rs_all), np.array(Us_all), np.array(Vs_all), v_top_all, selected_features), f)
    # with open('results/idx_tests.pkl', 'wb') as f:
    #     pickle.dump(idx_tests, f)
    with open(f'results/new-harm-mtimp/alpha_{alpha}_penaltyu_{penaltyu}_penaltyv_{penaltyv}_seed_{seed}@25.pkl', 'wb') as f:
        pickle.dump((reses, np.array(Rs_all), np.array(Us_all), Vs_all, v_top_all, selected_features), f)