import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from temp_norm import simulate, get_psth

def discounting_model_test(all_params, baseline=True, constrain_W=False, skip_sensory=False, suppress=True):
    # initialize
    t2 = []
    t3 = []
    t4 = []
    dt = 4 
    dt_psth = 20
    T = 644*dt_psth
    nt = T // dt
    soas = [1500, 3000, 6000]

    # get simulated reward peaks for each neuron
    for cur_params in all_params:
        if baseline and not constrain_W:
            tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, b, br = cur_params
        elif not baseline and constrain_W:
            tau3, sigma3, tauE3, tauS3, m, w1, scale_rwd, const_b = cur_params
            w4 = w3 = w2 = w1
            b = br = 0
        elif not baseline and not constrain_W and suppress:
            tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, const_b = cur_params
            b = br = 0
        elif not baseline and not constrain_W and not suppress:
            tau3, sigma3, tauE3, m, w1, w2, w3, w4, scale_rwd, const_b = cur_params
            tauS3 = 1
            b = br = 0
        else:
            raise ValueError("Invalid combination of baseline and constrain_W")
    
        predW = np.array([w1, w2, w3, w4, scale_rwd])

        for i, soa in enumerate(soas):
            # create trial stimulus
            stim = np.zeros((5, nt))
            cue_onset = 244*dt_psth//dt
            cue_offset = 294*dt_psth//dt
            rwd_onset = cue_onset + soa//dt
            rwd_offset = rwd_onset + 1000//dt
            stim[i+1, cue_onset:cue_offset] = 1 
            stim[4, rwd_onset:rwd_offset] = 1
    
            # simulate trial
            r3 = simulate(stim, predW, tau3=tau3, sigma3=sigma3, tauE3=tauE3, tauS3=tauS3, dt=dt, T=T, b=b, br=br, 
                          skip_sensory=skip_sensory, suppress=suppress) 
            yhat = m*r3
            if not baseline:
                yhat += const_b
    
            if i == 0:
                t2.append(yhat[rwd_onset:rwd_offset].max())
            elif i == 1:
                t3.append(yhat[rwd_onset:rwd_offset].max())
            elif i == 2:
                t4.append(yhat[rwd_onset:rwd_offset].max())

    # test differences between delays
    return wilcoxon(t2, t3, alternative="less").pvalue, wilcoxon(t3, t4, alternative="less").pvalue

def discounting_psth_test(raw_df, neurons, shuffle=False):
    # initialize
    t2 = []
    t3 = []
    t4 = []
    dt_psth = 20
    soas = [1500, 3000, 6000]

    # get reward peaks for each neuron
    for neuron in neurons:
        psth = get_psth(raw_df, neuron, shuffle=shuffle)
    
        for i, soa in enumerate(soas):
            rwd_peak = psth[i+1][244+soa//dt_psth:294+soa//dt_psth].max()
            if i == 0:
                t2.append(rwd_peak)
            elif i == 1:
                t3.append(rwd_peak)
            elif i == 2:
                t4.append(rwd_peak)

    # test differences between delays
    return wilcoxon(t2, t3, alternative="less").pvalue, wilcoxon(t3, t4, alternative="less").pvalue

def sousa_R2(all_params, raw_df, neurons, baseline=True, constrain_W=False, skip_sensory=False, suppress=True, shuffle=False):
    # initialize
    y = []
    yhat = []
    peaks_true = []
    peaks_pred = []
    dt = 4 
    dt_psth = 20
    T = 644*dt_psth
    nt = T // dt
    soas = [0, 1500, 3000, 6000]

    # get simulated reward peaks for each neuron
    for neuron_i, cur_params in enumerate(all_params):
        if baseline and not constrain_W:
            tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, b, br = cur_params
        elif baseline and constrain_W:
            tau3, sigma3, tauE3, tauS3, m, w1, scale_rwd, b, br = cur_params
            w4 = w3 = w2 = w1
        elif not baseline and constrain_W:
            tau3, sigma3, tauE3, tauS3, m, w1, scale_rwd, const_b = cur_params
            w4 = w3 = w2 = w1
            b = br = 0
        elif not baseline and not constrain_W and suppress:
            tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, const_b = cur_params
            b = br = 0
        elif not baseline and not constrain_W and not suppress:
            tau3, sigma3, tauE3, m, w1, w2, w3, w4, scale_rwd, const_b = cur_params
            tauS3 = 1
            b = br = 0
        else:
            raise ValueError("Invalid combination of baseline and constrain_W")
    
        predW = np.array([w1, w2, w3, w4, scale_rwd])

        psth = get_psth(raw_df, neurons[neuron_i], shuffle=shuffle, start=2000//dt_psth+6)

        b_psth = 0

        for soa_i in range(len(soas)):
            b_psth += psth[soa_i][144-1000//dt_psth:144].mean() / len(soas)

        for soa_i, soa in enumerate(soas):
            # create trial stimulus
            stim = np.zeros((5, nt))
            cue_onset = 244*dt_psth//dt
            cue_offset = 294*dt_psth//dt
            rwd_onset = cue_onset + soa//dt
            rwd_offset = rwd_onset + 1000//dt
            stim[soa_i, cue_onset:cue_offset] = 1 
            stim[4, rwd_onset:rwd_offset] = 1
    
            # simulate trial
            r3 = simulate(stim, predW, tau3=tau3, sigma3=sigma3, tauE3=tauE3, tauS3=tauS3, dt=dt, T=T, b=b, br=br, 
                          skip_sensory=skip_sensory, suppress=suppress)
            r3 *= m
            if not baseline:
                r3 += const_b
            b_hat = r3[cue_onset-1000//dt:cue_onset].mean()
            
            yhat.append(r3[2000//dt::dt_psth//dt])
            y.append(psth[soa_i])

            if (psth[soa_i][144:194].max() - b_psth)**2 > (psth[soa_i][144:194].min() - b_psth)**2:
                peaks_true.append(psth[soa_i][144:194].max() - b_psth)
                peaks_pred.append(r3[cue_onset:cue_offset].max() - b_hat)
            else:
                peaks_true.append(psth[soa_i][144:194].min() - b_psth)
                peaks_pred.append(r3[cue_onset:cue_offset].min() - b_hat)
            if (psth[soa_i][144+soa//dt_psth:194+soa//dt_psth].max() - b_psth)**2 > (psth[soa_i][144+soa//dt_psth:194+soa//dt_psth].min() - b_psth)**2:
                peaks_true.append(psth[soa_i][144+soa//dt_psth:194+soa//dt_psth].max() - b_psth)
                peaks_pred.append(r3[rwd_onset:rwd_offset].max() - b_hat)
            else:
                peaks_true.append(psth[soa_i][144+soa//dt_psth:194+soa//dt_psth].min() - b_psth)
                peaks_pred.append(r3[rwd_onset:rwd_offset].min() - b_hat)
    
    y = np.array(y).reshape(-1)
    yhat = np.array(yhat).reshape(-1)
    peaks_true = np.array(peaks_true)
    peaks_pred = np.array(peaks_pred)
    
    r2_tmp = r2_score(y, yhat)
    mse_tmp = np.mean((y - yhat)**2)
    aic_tmp = 2*all_params[0].shape[0] + y.shape[0]*np.log(mse_tmp)

    r2_shape = r2_score(peaks_true, peaks_pred)
    mse_shape = np.mean(peaks_true - peaks_pred)**2
    aic_shape = 2*all_params[0].shape[0] + peaks_true.shape[0]*np.log(mse_shape)
    
    return r2_tmp, aic_tmp, r2_shape, aic_shape
