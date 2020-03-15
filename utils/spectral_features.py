import numpy as np
from scipy.signal import welch
from scipy.integrate import simps

def mean_psd(signal,fs):
    freqs, psd = welch(signal,fs = fs)
    return np.mean(psd)

def a_b_relative_power(signal,fs):
    """
    https://www.ncbi.nlm.nih.gov/pubmed/8138380
    get alpha and beta relative power of the signal
    """
    freqs, psd = welch(signal,fs = fs)
    freq_res = freqs[1] - freqs[0]
    alpha_range = [8,13]
    beta_range = [14,30]
    idx_alpha = np.logical_and(freqs >= alpha_range[0], freqs <= alpha_range[1])
    idx_beta = np.logical_and(freqs >= beta_range[0], freqs <= beta_range[1])
    #compute absolute powers
    alpha_power = simps(psd[idx_alpha], dx=freq_res)
    beta_power = simps(psd[idx_beta], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    #compute relative powers
    alpha_rp = alpha_power/total_power
    beta_rp = beta_power/total_power

    return alpha_rp,beta_rp
