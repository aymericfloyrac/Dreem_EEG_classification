from eeg import EEG
from utils.hjorth import *
from utils.standard_features import *
from utils.spectral_features import *

class ManualEEG(EEG):

    def __init__(self,signal,label,indiv):
        super().__init__(signal,label,indiv)

        #hjorth features
        self.hjorth_extracted = False
        self.mobility = None
        self.activity = None
        self.complexity = None

    def extractFeatures(self):
        self.extractHjorthFeatures()
        self.extractStandardFeatures()
        self.extractSpectralFeatures()

        self.features = np.concatenate((self.mobility,self.activity,self.complexity,
                                        self.sd,self.maxi,self.mini,self.med,
                                        self.corr,self.sleeve_corr,
                                        self.mean_psd,self.alpha_rp,self.beta_rp))

        return self.features

    def extractHjorthFeatures(self):
        if self.clean_signal is None:
            self.clean()

        activity = np.apply_along_axis(getHjorthActivity,1,self.clean_signal)
        mobility = np.apply_along_axis(getHjorthMobility,1,self.clean_signal)
        complexity = np.apply_along_axis(getHjorthComplexity,1,self.clean_signal)


        self.hjorth_extracted = True
        self.mobility = mobility
        self.activity = activity
        self.complexity = complexity
        return activity,mobility,complexity

    def extractStandardFeatures(self):
        if self.clean_signal is None:
            self.clean()

        features = np.apply_along_axis(extract_std_ft,1,self.clean_signal)
        self.sd = [f[0] for f in features]
        self.maxi = [f[1] for f in features]
        self.mini = [f[2] for f in features]
        self.med = [f[3] for f in features]

        self.corr = extract_corr(self.clean_signal)
        self.sleeve_corr = extract_sleeve_corr(self.clean_signal)

        return self.sd,self.maxi,self.mini,self.med,self.corr,self.sleeve_corr

    def extractSpectralFeatures(self):
        if self.clean_signal is None:
            self.clean()

        mean_power = np.apply_along_axis(mean_psd,1,self.clean_signal,fs=self.sampling_freq)
        ab_powers = np.apply_along_axis(a_b_relative_power,1,self.clean_signal,fs = self.sampling_freq)

        self.mean_psd = mean_power
        self.alpha_rp = [p[0] for p in ab_powers]
        self.beta_rp = [p[1] for p in ab_powers]

        return self.mean_psd,self.alpha_rp,self.beta_rp
