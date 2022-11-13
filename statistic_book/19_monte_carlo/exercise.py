import numpy as np
import math
def ex1(num):
    total_res = []
    data = np.random.normal(loc = 0, scale = 1, size = k) # mean, scale = Standard deviation 
    data = data * data
    data = np.mean(data)
    return np.sqrt(2*math.pi) * data

if __name__ == '__main__':
    k = 1000
    result = ex1(k)
    print('the value is: ', result )
