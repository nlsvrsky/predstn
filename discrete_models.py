import numpy as np

def unexpec_resp_mse(x, rwds, responses):
    # error for unexpected response fit
    sigma, f = x
    hyp_resps = f*(rwds**.5 / (rwds**.5 + sigma**.5))
    return np.mean((responses - hyp_resps)**2)

def out_sub_mse(E, sigma, f, cues, rwds, responses):
    # error for output subtraction fit
    sub_resps = f*(rwds**.5 / (rwds**.5 + sigma**.5)) - E*cues
    return np.mean((responses - sub_resps)**2)

def out_sub_resp(sigma, f, E, cue, rwd):
    # response for output subtraction fit
    return f*(rwd**.5 / (rwd**.5 + sigma**.5)) - E*cue

def out_div_mse(E, sigma, f, cues, rwds, responses):
    # error for output division fit
    div_resps = f*(rwds**.5 / (rwds**.5 + sigma**.5)) * (1 / (cues*E**.5 + 1))
    return np.mean((responses - div_resps)**2)

def out_div_resp(sigma, f, E, cue, rwd):
    # response for output division fit
    return f*(rwd**.5 / (rwd**.5 + sigma**.5)) * (1 / (cue*E**.5 + 1))

def in_sub_mse(E, sigma, f, cues, rwds, responses):
    # error for input subtraction fit
    sub_resps = f*((rwds - E*cues)**.5 / ((rwds - E*cues)**.5 + sigma**.5))
    return np.mean((responses - sub_resps)**2)

def in_div_mse(E, sigma, f, cues, rwds, responses):
    # error for input division fit
    div_resps = f*(rwds**.5 / (rwds**.5 + sigma**.5 + cues*E**.5)) 
    return np.mean((responses - div_resps)**2)

def in_div_resp(sigma, f, E, cue, rwd):
    # response for input division fit (needed?)
    return f*(rwd**.5 / (rwd**.5 + sigma**.5 + cue*E**.5)) 


