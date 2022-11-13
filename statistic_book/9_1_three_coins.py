
import numpy as np
label_list = [1, 1, 0, 1, 0, 0, 1, 0, 1,1]
class em(object):
    def __init__(self, pai, p, q, label_list):
        self.pai = pai
        self.p = p
        self.q = q
        self.label_list = label_list
        self.u = [0] * len(self.label_list)
    
    def expectation(self):
        
        for i in range(len(self.u)):
            u_b = self.pai * self.p**(label_list[i]) * (1-self.p)**(1-label_list[i])
            u_c = self.pai * self.q**(label_list[i]) * (1-self.q)**(1-label_list[i])
            self.u[i] = u_b/(u_b + u_c)

    
    def maxi(self):
        u_array = np.array(self.u)
        label_array = np.array(self.label_list)
        self.pai = np.average(self.u)
        self.p = np.dot(u_array, label_array)/sum(self.u)
        self.q = np.dot(1-u_array, label_array)/sum(self.u)
    
    def fit(self, num_iter):
        for i in range(num_iter):
            self.expectation()
            self.maxi()
            print('pai, p, q', self.pai, self.p , self.q)

a = em(pai = 0.46, p =0.55, q = 0.67, label_list= label_list)
a.fit(5)
        