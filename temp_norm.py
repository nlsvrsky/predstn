import numpy as np
import pandas as pd

def simulate(stim, 
             predW,
             dt=2, 
             T=12*1000,  
             tau1=3, 
             sigma1=.5, 
             tauE1=.015,
             tauS1=.01,
             t1r=11.25,
             E1r=70,
             S1r=70,
             tau2=7.5,
             sigma2=.1,
             tauE2=5,
             tauS2=10,
             sigma2r=2,
             E2r=1,
             S2r=1,
             tau3=5, 
             sigma3=2, 
             tauE3=10, 
             tauS3=10, 
             p=1.5,
             b=.1, 
             br=.1,
             tau_scale=10, 
             skip_sensory=False,
             suppress=True):

    # tau parameters were scaled numerically for the optimization
    tau1 *= tau_scale
    tauE1 *= tau_scale
    tauS1 *= tau_scale
    tau2 *= tau_scale
    tauE2 *= tau_scale
    tauS2 *= tau_scale
    tau3 *= tau_scale
    tauE3 *= tau_scale
    tauS3 *= tau_scale
    
    # initialize 
    nt = T // dt
    tempWE1 = np.exp(-np.arange(0, nt)*dt/(tauE1)) * dt/(tauE1)  
    tempWS1 = np.exp(-np.arange(0, nt)*dt/(tauS1)) * dt/(tauS1) 
    tempWE1_r = np.exp(-np.arange(0, nt)*dt/(tauE1*E1r)) * dt/(tauE1*E1r) 
    tempWS1_r = np.exp(-np.arange(0, nt)*dt/(tauS1*S1r)) * dt/(tauS1*S1r)
    tempWE2 = np.exp(-np.arange(0, nt)*dt/(tauE2)) * dt/(tauE2)  
    tempWS2 = np.exp(-np.arange(0, nt)*dt/(tauS2)) * dt/(tauS2) 
    tempWE2_r = np.exp(-np.arange(0, nt)*dt/(tauE2*E2r)) * dt/(tauE2*E2r) 
    tempWS2_r = np.exp(-np.arange(0, nt)*dt/(tauS2*S2r)) * dt/(tauS2*S2r)
    tempWE3 = np.exp(-np.arange(0, nt)*dt/tauE3) * dt/tauE3
    tempWS3 = np.exp(-np.arange(0, nt)*dt/tauS3) * dt/tauS3
    drive1, d1, s1, f1, r1 = np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt))
    drive2, d2, s2, f2, r2 = np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt)), np.zeros((5, nt))
    drive3, d3, s3, f3, r3 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    sensW = np.array([.1, .1, .1, .1, 1])
    
    # simulate
    for i in range(1, nt):
        if not skip_sensory:
            # sensory layer 1
            drive1[:, i] = stim[:, i]**p
            d1[:4, i] = np.sum(drive1[:4, :i] * tempWE1[i-1::-1], axis=1) 
            d1[4, i] = np.sum(drive1[4, :i] * tempWE1_r[i-1::-1]) 
            s1[:4, i] = np.sum(np.abs(d1[:4, :i]) * tempWS1[i-1::-1], axis=1) 
            s1[4, i] = np.sum(np.abs(d1[4, :i]) * tempWS1_r[i-1::-1]) 
            f1[:4, i] = d1[:4, i] / (np.abs(s1[:4, i]) + (sigma1)**p) 
            f1[4, i] = d1[4, i] / (np.abs(s1[4, i]) + (sigma1)**p) 
            r1[:4, i] = r1[:4, i-1] + dt/(tau1)*(-r1[:4, i-1]+f1[:4, i]) 
            r1[4, i] = r1[4, i-1] + dt/(tau1*t1r)*(-r1[4, i-1]+f1[4, i])  
    
            # sensory layer 2
            drive2[:, i] = (r1[:, i] * sensW)**p
            d2[:4, i] = np.sum(drive2[:4, :i] * tempWE2[i-1::-1], axis=1) 
            d2[4, i] = np.sum(drive2[4, :i] * tempWE2_r[i-1::-1]) 
            s2[:4, i] = np.sum(np.abs(d2[:4, :i]) * tempWS2[i-1::-1], axis=1) 
            s2[4, i] = np.sum(np.abs(d2[4, :i]) * tempWS2_r[i-1::-1]) 
            f2[:4, i] = (d2[:4, i] + b) / (np.abs(s2[:4, i]).sum() + (sigma2)**p) 
            f2[4, i] = (d2[4, i] + br) / (np.abs(s2[4, i]).sum() + (sigma2*sigma2r)**p) 
            r2[:4, i] = r2[:4, i-1] + dt/(tau2)*(-r2[:4, i-1]+f2[:4, i]) 
            r2[4, i] = r2[4, i-1] + dt/(tau2)*(-r2[4, i-1]+f2[4, i]) 

        # reward layer
        if not skip_sensory:
            drive3[i] = max(0, predW @ r2[:, i])**p 
        else:
            drive3[i] = max(0, predW @ stim[:, i]**p)
        d3[i] = np.sum(drive3[:i] * tempWE3[i-1::-1]) 
        s3[i] = np.sum(np.abs(d3[:i]) * tempWS3[i-1::-1])
        f3[i] = (d3[i]) / (int(suppress)*s3[i] + sigma3**p) 
        r3[i] = r3[i-1] + dt/tau3*(-r3[i-1]+f3[i]) 

    return r3

def wrap_simulate_sousa(args, psth, plot=False, retrieve=False, cue_only=False, rwd_only=False, baseline=True, constrain_W=False, skip_sensory=False, suppress=True):
    if baseline and not constrain_W:
        tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, b, br = args 
    elif not baseline and constrain_W:
        tau3, sigma3, tauE3, tauS3, m, w1, scale_rwd, const_b = args 
        w4 = w3 = w2 = w1
        b = br = 0
    elif not baseline and not constrain_W and suppress:
        tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, const_b = args 
        b = br = 0
    elif not baseline and not constrain_W and not suppress:
        tau3, sigma3, tauE3, m, w1, w2, w3, w4, scale_rwd, const_b = args 
        tauS3 = 1
        b = br = 0
    else:
        raise ValueError("Invalid combination of baseline and constrain_W")

    # initialize
    dt = 4 
    dt_psth = 20
    T = 644*dt_psth
    nt = T // dt
    predW = np.array([w1, w2, w3, w4, scale_rwd])
    
    if retrieve:
        yhats_retrieve = []
    if plot:
        colors = ["red", "orange", "green", "blue"]
        
    soas = [0, 1500, 3000, 6000]
    error = 0

    b_psth = 0

    for i in range(len(soas)):
        b_psth += psth[i][244-1000//dt_psth:244].mean() / len(soas)
    
    for i, soa in enumerate(soas):
        # create trial stimulus
        stim = np.zeros((5, nt))
        cue_onset = 244*dt_psth//dt
        cue_offset = 294*dt_psth//dt
        rwd_onset = cue_onset + soa//dt
        rwd_offset = rwd_onset + 1000//dt
        stim[i, cue_onset:cue_offset] = 1 
        stim[4, rwd_onset:rwd_offset] = 1

        # simulate trial
        r3 = simulate(stim, predW, tau3=tau3, sigma3=sigma3, tauE3=tauE3, tauS3=tauS3, dt=dt, T=T, b=b, br=br, 
                      skip_sensory=skip_sensory, suppress=suppress) 
        yhat = m*r3
        if not baseline:
            yhat += const_b
        b_hat = yhat[cue_onset-1000//dt:cue_onset].mean()

        # save responses if needed 
        if retrieve:
            yhats_retrieve.append(yhat)

        # prox objective: min and max peaks + total MSE
        if psth is not None: 
            if not rwd_only:
                error += np.mean((psth[i][0:] - yhat[0::dt_psth//dt])**2)

            if not rwd_only:
                if (psth[i][244:294].max() - b_psth)**2 > (psth[i][244:294].min() - b_psth)**2 or not baseline: 
                    error += ((psth[i][244:294].max()-b_psth) - (yhat[cue_onset:cue_offset].max()-b_hat))**2 
                else:
                    error += ((psth[i][244:294].min()-b_psth) - (yhat[cue_onset:cue_offset].min()-b_hat))**2 

            if (psth[i][244+soa//dt_psth:294+soa//dt_psth].max() - b_psth)**2 > (psth[i][244+soa//dt_psth:294+soa//dt_psth].min() - b_psth)**2 or not baseline:
                error += ((psth[i][244+soa//dt_psth:294+soa//dt_psth].max()-b_psth) - (yhat[rwd_onset:rwd_offset].max()-b_hat))**2 
            else:
                error += ((psth[i][244+soa//dt_psth:294+soa//dt_psth].min()-b_psth) - (yhat[rwd_onset:rwd_offset].min()-b_hat))**2 
    
    if retrieve:
        return yhats_retrieve, error
    return error

def wrap_simulate_eshel(args, testcue=1, testrwd_dur=0):
    tau3, sigma3, tauE3, tauS3, m, w1, w2, w3, w4, scale_rwd, b, br = args

    # initialize
    dt = 4
    dt_psth = 20
    T = 644*dt_psth
    nt = T // dt
    predW = np.array([w1, w2, w3, w4, scale_rwd])
    soa = 500

    # simulate trial
    stim = np.zeros((5, nt))
    cue_onset = 244*dt_psth//dt
    cue_offset = 294*dt_psth//dt
    rwd_onset = cue_offset + soa//dt
    rwd_offset = rwd_onset + testrwd_dur//dt
    stim[1, cue_onset:cue_offset] = testcue 
    stim[4, rwd_onset:rwd_offset] = 1 
    r3 = simulate(stim, predW, tau3=tau3, sigma3=sigma3, tauE3=tauE3, tauS3=tauS3, dt=dt, T=T, b=b, br=br) 
    yhat = m*r3

    return np.array(yhat)#.reshape(-1)
            

def get_psth(raw_df, neuron, r=4.5, get_n=False, start=6, shuffle=False):
    if get_n:
        trial_counts = []
        
    delay6 = raw_df[(raw_df["Distribution reward ID"] == 0) & (raw_df["Amount reward"] == r) & (raw_df["Delay reward"] == 6) & (raw_df["Is photo ided"] == 1)]
    delay6 = delay6[delay6["Neuron id"] == neuron]
    psth6 = delay6[delay6.columns[(delay6.columns.str.contains("PSTH")) & (delay6.columns.str.contains("aligned to cue"))]]
    if get_n:
        trial_counts.append(psth6.shape[0])
    psth6 = psth6.mean().iloc[start:]
    
    delay3 = raw_df[(raw_df["Distribution reward ID"] == 0) & (raw_df["Amount reward"] == r) & (raw_df["Delay reward"] == 3) & (raw_df["Is photo ided"] == 1)]
    delay3 = delay3[delay3["Neuron id"] == neuron]
    psth3 = delay3[delay3.columns[(delay3.columns.str.contains("PSTH")) & (delay3.columns.str.contains("aligned to cue"))]]
    if get_n:
        trial_counts.append(psth3.shape[0])
    psth3 = psth3.mean().iloc[start:]
    
    delay15 = raw_df[(raw_df["Distribution reward ID"] == 0) & (raw_df["Amount reward"] == r) & (raw_df["Delay reward"] == 1.5) & (raw_df["Is photo ided"] == 1)]
    delay15 = delay15[delay15["Neuron id"] == neuron]
    psth15 = delay15[delay15.columns[(delay15.columns.str.contains("PSTH")) & (delay15.columns.str.contains("aligned to cue"))]]
    if get_n:
        trial_counts.append(psth15.shape[0])
    psth15 = psth15.mean().iloc[start:]
    
    delay0 = raw_df[(raw_df["Distribution reward ID"] == 0) & (raw_df["Amount reward"] == r) & (raw_df["Delay reward"] == 0) & (raw_df["Is photo ided"] == 1)]
    delay0 = delay0[delay0["Neuron id"] == neuron]
    if delay0.shape[0] == 0:
        print("Missing 0 delay for neuron")
        return None
    psth0 = delay0[delay0.columns[(delay0.columns.str.contains("PSTH")) & (delay0.columns.str.contains("aligned to cue"))]]
    if get_n:
        trial_counts.append(psth0.shape[0])
    psth0 = psth0.mean().iloc[start:]

    psth = [psth0, psth15, psth3, psth6]
    
    if shuffle:
        return shuffle_psth(psth, neuron, start=start)
    if get_n:
        return psth, reverse(trial_counts)
    return psth

def shuffle_psth(psth, neuron, start=6):
    soas = [0, 1500//20, 3000//20, 6000//20]
    np.random.seed(neuron) # use the neuron as the random seed so that a given neuron is shuffled the same way
    
    new_orders = np.concatenate([[0], np.random.permutation([1, 2, 3])])
    while np.all(new_orders == np.arange(4)):
        new_orders = np.concatenate([[0], np.random.permutation([1, 2, 3])])
    
    psth_shuff = []
    for i_to in range(4):
        i_from = new_orders[i_to]
        splice_to = 250 - start + soas[i_to]
        splice_from = 250 - start + soas[i_from]
        psth_shuff.append(np.concatenate((psth[i_to][:splice_to], psth[i_from][splice_from:splice_from+100], psth[i_to][splice_to+100:])))

    return psth_shuff
    