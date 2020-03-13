from eeg import EEG
from utils.hjorth import *

class ManualEEG(EEG):

    def __init__(self,signal,label,indiv):
        super().__init__(signal,label,indiv)

        #hjorth features
        self.hjorth_extracted = False
        self.mobility = None
        self.activity = None
        self.complexity = None

    def extract_features(self):
        return None

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

    def hjorthViz(self):
        if not self.hjorth_extracted:
            self.extractHjorthFeatures()
        if label == 0:
            c='b'
        else:
            c='m'

        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.hist(self.mobility,c=c)
        ax1.set_title('Mobility')
        ax2.hist(self.activity,c=c)
        ax2.set_title('Activity')
        ax3.hist(self.complexity,c=c)
        ax3.set_title('Complexity')
