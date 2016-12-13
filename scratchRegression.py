from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6],dtype=np.float64)
ys = np.array([5,6,7,8,9,12],dtype=np.float64)

def create_dataset(hm, variance, step=2 , correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)

    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
    m = ((mean(xs)*mean(ys) -mean(xs*ys))/
         (mean(xs)**2 - mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m,b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficent_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    square_error_regr = squared_error(ys_orig,ys_line)
    square_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (square_error_regr/square_error_y_mean)



m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]

r_squared = coefficent_of_determination(ys,regression_line)
print(r_squared)

predict_x = 8
predict_y = (m*predict_x) + b


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs,regression_line)
plt.show()

