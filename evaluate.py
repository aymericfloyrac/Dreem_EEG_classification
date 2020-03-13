from sklearn.metrics import accuracy,recall,precision


def evaluate(model):

    path_to_data = './data/'
    
    model.fit()
