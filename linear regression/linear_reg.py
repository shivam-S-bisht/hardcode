# y = mx + c
import pandas as pd

def start():

    #importing dataset (dataframe)
    dataset = pd.read_csv('data.csv')

    #array dataset
    dataset = dataset.iloc[:, :].values

    #initialising parameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    n_iter = 1000
    print(cost(initial_b, initial_m, dataset))
    [b, m] = gradient_descend(dataset, learning_rate, initial_b, initial_m, n_iter)
    
    print(cost(b, m, dataset))


#error function
def cost(b, m, dataset):
    total_error = 0
    N = int(len(dataset))
    for i in range(N):
        x = dataset[i, 0]
        y = dataset[i, 1]

        #squaring the error
        total_error += (y - (m * x + b)) ** 2

    return total_error / N    

#dradient function
def gradient_descend(dataset, learning_rate, starting_b, starting_m, n_iter):
    b = starting_b
    m = starting_m
    for i in range(n_iter):
        b, m = gradient_step(b, m, dataset, learning_rate)
    
    return [b, m]


def gradient_step(b_current, m_current, dataset, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = int(len(dataset))
    for i in range(N):
        x = dataset[i, 0]
        y = dataset[i, 1]

        #converging the cost function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    #new parameters
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return (new_b, new_m)


if __name__ == '__main__':
    start()