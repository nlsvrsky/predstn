import numpy as np
import matplotlib.pyplot as plt
from temp_norm import wrap_simulate_sousa, get_psth

def plot_all_neurons(params, neurons, raw_df, filename, avg_filename, avg_description=" (all neurons)", 
                     baseline=True, constrain_W=False, skip_sensory=False, suppress=True, shuffle=False):
    dt_psth = 20
    dt = 4
    
    cue_time = 5000 - 2000 - 6*dt_psth

    # individual neurons
    fig, axs = plt.subplots(7, 8, figsize=(30, 15), constrained_layout=True)
    
    for i in range(len(neurons)):
        if i < 3:
            row = (i + 1) // 4
            col = (i + 1) % 4
        elif i < 6:
            row = (i + 2) // 4
            col = (i + 2) % 4
        else:
            row = (i + 3) // 4
            col = (i + 3) % 4
    
        # PSTH
        psth0, psth15, psth3, psth6 = get_psth(raw_df, neurons[i], start=2000//dt_psth+6, shuffle=shuffle)
    
        if i == 0:
            avg_psth0, avg_psth15, avg_psth3, avg_psth6 = psth0, psth15, psth3, psth6
        else:
            avg_psth0 += psth0
            avg_psth15 += psth15
            avg_psth3 += psth3
            avg_psth6 += psth6
            
        axs[row, 2*col].plot(np.arange(0, psth0.shape[0]), psth0, color="#f7b0a8")
        axs[row, 2*col].plot(np.arange(0, psth15.shape[0]), psth15, color="#ed7042")
        axs[row, 2*col].plot(np.arange(0, psth3.shape[0]), psth3, color="#db0202")
        axs[row, 2*col].plot(np.arange(0, psth6.shape[0]), psth6, color="#870101")
        
    
        # temporal normalization response
        #yhats, error = wrap_simulate_sousa(results[i]["min_res"].x, get_psth(raw_df, neurons[i]), retrieve=True)  
        yhats, _ = wrap_simulate_sousa(params[i], get_psth(raw_df, neurons[i], shuffle=shuffle), 
                                       retrieve=True, baseline=baseline, constrain_W=constrain_W, skip_sensory=skip_sensory, suppress=suppress)  
        yhat1 = np.array(yhats[0])[2000//dt:]
        yhat2 = np.array(yhats[1])[2000//dt:]
        yhat3 = np.array(yhats[2])[2000//dt:]
        yhat4 = np.array(yhats[3])[2000//dt:]
    
        if i == 0:
            avg_yhat1, avg_yhat2, avg_yhat3, avg_yhat4 = yhat1, yhat2, yhat3, yhat4
        else:
            avg_yhat1 += yhat1
            avg_yhat2 += yhat2
            avg_yhat3 += yhat3
            avg_yhat4 += yhat4
        
        axs[row, 2*col+1].plot(np.arange(0, yhat1.shape[0]), yhat1, color="#f7b0a8")
        axs[row, 2*col+1].plot(np.arange(0, yhat2.shape[0]), yhat2, color="#ed7042")
        axs[row, 2*col+1].plot(np.arange(0, yhat3.shape[0]), yhat3, color="#db0202")
        axs[row, 2*col+1].plot(np.arange(0, yhat4.shape[0]), yhat4, color="#870101")
    
        # axis limits
        upper = np.concatenate([psth0, psth15, psth3, psth6, yhat1, yhat2, yhat3, yhat4]).max() + 1
        lower = np.concatenate([psth0, psth15, psth3, psth6, yhat1, yhat2, yhat3, yhat4]).min() - 1
        axs[row, 2*col].set_ylim(lower, upper)
        axs[row, 2*col+1].set_ylim(lower, upper)
        axs[row, 2*col].set_title("Neuron " + str(i + 1) + " data")
        axs[row, 2*col+1].set_title("Neuron " + str(i + 1) + " fit")
    
        if row < 4:
            axs[row, 2*col].set_xticks([])
            axs[row, 2*col+1].set_xticks([])
        else:
            data_times = np.array([cue_time, cue_time + 1500, cue_time + 3000, cue_time + 6000]) // dt_psth
            fit_times = np.array([cue_time, cue_time + 1500, cue_time + 3000, cue_time + 6000]) // dt 
            axs[row, 2*col].set_xticks(data_times, labels=["0", "1.5", "3.0", "6.0"])
            axs[row, 2*col+1].set_xticks(fit_times, labels=["0", "1.5", "3.0", "6.0"])
    
    # leave space for average plots
    for row in range(3):
        for col in range(2):
            axs[row, col].remove()
    
    avg_psth0 /= len(neurons)
    avg_psth15 /= len(neurons)
    avg_psth3 /= len(neurons)
    avg_psth6 /= len(neurons)
    avg_yhat1 /= len(neurons)
    avg_yhat2 /= len(neurons)
    avg_yhat3 /= len(neurons)
    avg_yhat4 /= len(neurons)
    
    fig.supxlabel("Time from cue onset (seconds)", fontsize=18)
    fig.supylabel("Firing rate (spikes per second)", fontsize=18)
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    # average 
    fig, axs = plt.subplots(2, 1, figsize=(4, 5), constrained_layout=True)
    axs[0].plot(np.arange(0, avg_yhat1.shape[0]), avg_yhat1, color="#f7b0a8")
    axs[0].plot(np.arange(0, avg_yhat2.shape[0]), avg_yhat2, color="#ed7042")
    axs[0].plot(np.arange(0, avg_yhat3.shape[0]), avg_yhat3, color="#db0202")
    axs[0].plot(np.arange(0, avg_yhat4.shape[0]), avg_yhat4, color="#870101")
    axs[0].set_title("Average fit" + avg_description)
    axs[0].set_xticks([])
    axs[0].set_yticks([3, 8, 13, 18])
    axs[0].set_ylim(2, 19)
    
    axs[1].plot(np.arange(0, avg_psth0.shape[0]), avg_psth0, color="#f7b0a8")
    axs[1].plot(np.arange(0, avg_psth15.shape[0]), avg_psth15, color="#ed7042")
    axs[1].plot(np.arange(0, avg_psth3.shape[0]), avg_psth3, color="#db0202")
    axs[1].plot(np.arange(0, avg_psth6.shape[0]), avg_psth6, color="#870101")
    axs[1].set_title("Average data" + avg_description)
    axs[1].set_xticks(data_times, labels=["0", "1.5", "3.0", "6.0"])
    axs[1].set_yticks([3, 8, 13, 18])
    axs[1].set_ylim(2, 19)
    
    plt.savefig(avg_filename, bbox_inches="tight")
    plt.show()