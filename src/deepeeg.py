from eeg import EEG

class DeepEEG(EEG):

    def __init__(self,signal,label,indiv):
        super().__init__(signal,label,indiv)

    def extract_features(self):
        return None 
