import numpy as np
import matplotlib.pyplot as plt
import polyfit as pf
from sklearn.model_selection import KFold as ks
import importlib
importlib.reload(ks)

# Loading training and test data
x_train = np.loadtxt('Data/x_train.csv',delimiter=',')
y_train = np.loadtxt('Data/y_train.csv',delimiter=',')
x_test = np.loadtxt('Data/x_test.csv',delimiter=',')
y_test = np.loadtxt('Data/y_test.csv',delimiter=',')

# Plotting data purely for verification
plt.plot(x_train,y_train,'k.',x_test,y_test,'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend({'Training','Testing'})
plt.show()

# Fitting model
deg = 2
beta = pf.fit(x_train,y_train,deg)

# Computing training error
y_train_pred = pf.predict(x_train,beta)
err = pf.rmse(y_train,y_train_pred)
print('Training Error = {:2.3}'.format(err))

# Computing test error
y_test_pred = pf.predict(x_test,beta)
err = pf.rmse(y_test,y_test_pred)
print('Test Error = {:2.3}'.format(err))

# Plotting fitted model
x = np.linspace(0,1,100)
y = pf.predict(x,beta)
plt.plot(x,y,'b-',x_train,y_train,'ks',x_test,y_test,'rs')
plt.legend(['Prediction','Training Points','Test Points'])
plt.show()



# This is the total number of folds
K = 10

# This is the function that needs to be modified within the specified block in the script.
# Note that the last entry is the current k-fold splid to be returned. That is, passing a
# value of 0 will return the first fold split with the first fold as the validation set.
x_preval, y_preval, x_val, y_val = ks.split(x_train,y_train,K,0)

print('Dimensions of Validation Set: {}'.format(x_val.shape))
print('Dimensions of Pre-Validation Set: {}'.format(x_preval.shape))