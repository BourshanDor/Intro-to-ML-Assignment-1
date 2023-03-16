import math
import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import t
# from scipy.stats import poisson



def main(): 

    print("Generate an N x n matrix of samples from Bernoulli(p) ")
    # p = float(input("Enter bernoulli parameter:\n"))
    # N = int(input("Enter N:\n"))
    # n = int(input("Enter n:\n"))
    p = 0.5 
    N = 200000 
    n = 20 
    bernoulli_matrix = create_bernoullii_matrix(p,N,n) 
    print(bernoulli_matrix)
    result_mean = [np.mean(bernoulli_matrix[i]) for i in range(n)]
    print(result_mean)
    epsilon_random, y_axis = empirial_epsilon(result_mean, 50)
    hoeffding_bound = [2*math.pow(math.e, (-2)*N*(math.pow(epsilon_random[i],2))) for i in range(len(epsilon_random))] 
    plot_care(epsilon_random, y_axis, hoeffding_bound) 

    

def empirial_epsilon(result_mean, num_of_epsilon,) : 
    epsilon_random = np.linspace(0,1,num_of_epsilon)
    y_axis = [] 
    for j in range(num_of_epsilon) : 
        y= [1 if (abs(result_mean[i] - 0.5 ) > epsilon_random[j]) else 0  for i in range(len(result_mean))] 
        y_axis.append(np.mean(y)) 
    return epsilon_random, y_axis

def create_bernoullii_matrix(p: int,N:int ,n:int ) -> np.array:
    return np.random.binomial(1,p, size=(n,N))

def plot_care(x,y, hoeffding_bound) : 
    plt.plot(x, y, label="Empirial")
    plt.plot(x, hoeffding_bound, label="Hoeffding bound")
    plt.title("Hoeffding bound")
    plt.xlabel("epsilon")
    plt.ylabel("probability |X - 1/2| > epsilon")
    plt.show()
    
        



#     x = np.linspace(mean - 3*STD, mean + 3*std, 100)
#     plt.plot(x, stats.norm.pdf(x, mean, std),label="Norm")

#     plt.legend()
#     plt.show()

# def q3e6(grades, mean): 
#     sum = 0 
#     for x in grades: 
#         sum +=  (x-mean)**2
#     print(sum/14)

# def clalPPF ():
#     print(stats.norm.ppf(0.975)*10*(1/math.sqrt(15)))



if __name__ == "__main__":
    main()