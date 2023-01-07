
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import pandas as pd


# # # from sklearn.model_selection import train_test_split
# # # from sklearn.linear_model import LinearRegression

# # # N = 300 # 100 samples
# # # x = np.linspace(-3, 3, N) # coordinates
# # # noise_sample = np.random.RandomState(20)  # constant random value
# # # # growing sinusoid with random fluctuation
# # # sine_wave = x + np.sin(4*x) + noise_sample.uniform(N)
# # # # plt.plot(x, sine_wave, 'o');
# # # # plt.show()

# # # # convert features in 2D format i.e. list of list
# # # # print('Before: ', x.shape)
# # # features = x[:, np.newaxis]
# # # # print('After: ', features.shape)

# # # # uncomment below line to see the differences
# # # # print(x)
# # # # print(features)

# # # # save sine wave in variable 'targets'
# # # targets = sine_wave


# # # # split the training and test data
# # # train_features, test_features, train_targets, test_targets = train_test_split(
# # #         features, targets,
# # #         train_size=0.8,
# # #         test_size=0.2,
# # #         # random but same for all run, also accuracy depends on the
# # #         # selection of data e.g. if we put 10 then accuracy will be 1.0
# # #         # in this example
# # #         random_state=23,
# # #         # keep same proportion of 'target' in test and target data
# # #         # stratify=targets  # can not used for single feature
# # #     )

# # # # training using 'training data'
# # # regressor = LinearRegression()
# # # regressor.fit(train_features, train_targets) # fit the model for training data

# # # # predict the 'target' for 'training data'
# # # prediction_training_targets = regressor.predict(train_features)

# # # # note that 'score' uses 'feature and target (not predict_target)'
# # # # for scoring in Regression
# # # # whereas 'accuracy_score' uses 'features and predict_targets'
# # # # for scoring in Classification
# # # self_accuracy = regressor.score(train_features, train_targets)
# # # print("Accuracy for training data (self accuracy):", self_accuracy)

# # # # predict the 'target' for 'test data'
# # # prediction_test_targets = regressor.predict(test_features)
# # # test_accuracy = regressor.score(test_features, test_targets)
# # # print("Accuracy for test data:", test_accuracy)

# # # # plot the predicted and actual target for test data
# # # plt.scatter(test_features,test_targets,color='r')
# # # plt.plot(test_features,prediction_test_targets)
# # # # plt.plot(test_targets,prediction_test_targets, '-*')
# # # print(test_targets.shape,type(test_targets))
# # # # plt.plot(test_targets, '-o' )
# # # plt.show()

# # n_samples = 200
# # x = np.linspace(-10, 10, n_samples) # coordinates
# # noise_sample = np.random.normal(0,0.5,n_samples)
# # sine_wave = x + np.sin(4*x) + noise_sample
# # y = sine_wave

# # points = 12
# # names = []
# # for i in range(points):
# #   names.append("a"+str(i))
# # ds = []
# # target = []
# # for i in range(points,len(x)-1):
# #   ds.append(y[i-points:i])
# #   target.append(y[i]) 
# # df = pd.DataFrame(np.array(ds),columns=names) 
# # dfy = df 
# # dfy["target"] = target

# # # print(dfy[names])
# # # print(dfy[names].to_numpy())
# # # print(dfy["target"].to_numpy())

# # new_targets = dfy["target"].to_numpy()
# # # new_targets = new_targets[:, np.newaxis]

# # new_X = dfy[names].to_numpy()

# # for x in new_X[:2]:
# #   print(np.array([1.0] + list(x)))
# #   print(x)
# #   # print(x)

# # # X_with_bias = [[1] + x for x in new_X]
# # # print(new_X[:10])
# # # print(X_with_bias[:10])

# # print(new_X.shape,new_targets.shape)

# import numpy as np
# import matplotlib.pyplot as plt

# # creating data
# mean = np.array([5.0, 6.0])
# cov = np.array([[1.0, 0.95], [0.95, 1.2]])
# data = np.random.multivariate_normal(mean, cov, 8000)

# # visualising data
# plt.scatter(data[:500, 0], data[:500, 1], marker='.')
# plt.show()

# # train-test-split
# data = np.hstack((np.ones((data.shape[0], 1)), data))

# split_factor = 0.90
# split = int(split_factor * data.shape[0])

# X_train = data[:split, :-1]
# y_train = data[:split, -1].reshape((-1, 1))
# X_test = data[split:, :-1]
# y_test = data[split:, -1].reshape((-1, 1))

# print(X_train.shape,y_train.shape)
# print(X_train[:3])
# print("---")
# print(y_train[:3])

# # print(& quot
# #     Number of examples in training set= % d & quot
# #      % (X_train.shape[0]))
# # print(& quot
# #   Number of examples in testing set= % d & quot
# #   % (X_test.shape[0]))

import numpy as np
import matplotlib.pyplot as plt
# a = np.array([[10,20,111],[30,40,111],[50,60,111],[70,80,111]])

b = np.array([[1],[2],[3]])

# c = np.array([])

# print(c.shape[0])
# print(b[:,1:])

a  = np.array([1,2,3,4,5,6])

# print(a.reshape(-1,1))

print(b / b.max())

plt.plot(a)
plt.show()