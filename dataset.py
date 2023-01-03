from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def simple_dataset(n_samples = 200,n_features = 1,noise = 20):
    # Simple dataset
    
    X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, n_informative=1, noise=noise, random_state=37)
    # fig = plt.figure(figsize=(4, 4))

    # plt.xlabel("x (feature)")
    # plt.ylabel("y (output)")
    # plt.title("Synthetic data set")
    # plt.scatter(X, y)
    # plt.show()

    # print(type(X),X,type(y))
    return X,y


def complex_dataset(n_samples = 300):
    # Complex dataset
    
    x = np.linspace(-10, 10, n_samples) # coordinates
    noise_sample = np.random.normal(0,0.5,n_samples)
    sine_wave = x + np.sin(4*x) + noise_sample
    
    # plt.plot(x, sine_wave, 'o');
    # plt.show()

    

    return x,sine_wave


def remodel_complex_dataset(n_samples = 300):
    x = np.linspace(-10, 10, n_samples) # coordinates
    noise_sample = np.random.normal(0,0.5,n_samples)
    sine_wave = x + np.sin(4*x) + noise_sample
    y = sine_wave

    points = 12
    names = []
    for i in range(points):
        names.append("a"+str(i))
    ds = []
    target = []
    for i in range(points,len(x)-1):
        ds.append(y[i-points:i])
        target.append(y[i]) 
    df = pd.DataFrame(np.array(ds),columns=names) 
    dfy = df 
    dfy["target"] = target

    new_X = dfy[names].to_numpy()
    new_targets = dfy["target"].to_numpy()


    with_bias_X = []

    for x in new_X:
        with_bias_X.append(np.array([1.0] + list(x)))
    
    with_bias_X = np.array(with_bias_X)

    return with_bias_X,new_targets