import numpy as np
import matplotlib.pyplot as plt
import matplotlib 


def jit_plus_server(parameters):
    x = np.array(parameters)
    d = len(x)
    x1 = x - 0.15*np.ones(shape=(d,))
    x2 = x - 0.85*np.ones(shape=(d,))
    cpu_time = 2-np.exp(-10*x1**2) - 0.5*np.exp(-10*x2**2)
    return cpu_time.mean() + 0.005*np.random.normal()

class BayesianOptimizer:
    def __init__(self, num_parameters):
        self.num_parameters =num_parameters
        self.parameters = []
        self.measurements = []
        self.xo = np.array([0.5]*num_parameters)
    
    def ask(self):
        if len(self.measurements) == 0:
            return self.xo
        return self.new_parameter()
    
    def new_parameter(self):
        gpr = GPR4(
            self.parameters,
            self.measurements,
            sigma = 0.15,
        )
        new_vector = random_search(gpr, self.num_parameters)
        return new_vector
    
    def tell(self, parameter, measurement):
        self.parameters.append(parameter)
        self.measurements.append(measurement)

    def plot(self, x, y, title="Iterations vs CPU Time"):
        matplotlib.rcParams.update({'font.size': 24})
        fig, ax  = plt.subplots(figsize=(11,9))
        plt.rc('figure', titlesize=30)  
        ax.plot(x, y, '-.', linewidth=4.0)
        if title is not None:
            ax.set_title(title )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("CPU Time")
        plt.show()
        plt.close()



def random_search(gpr, num_parameters, num_iterations=200):
    step_size = 0.1
    x_current = np.random.normal(size=num_parameters)
    x_current, lcb_current = evaluate(gpr, x_current)
    for _ in range(num_iterations):
        x_test = (x_current + 
                  step_size*np.random.normal(size=num_parameters))
        x_test, lcb_test = evaluate(gpr, x_test)
        if lcb_test < lcb_current:
            lcb_current = lcb_test
            x_current = x_test
    return x_current


def evaluate(gpr, x):
    x = np.mod(x, 1)
    y, sigma_y = gpr.estimate(x)
    lcb = y - sigma_y
    return x, lcb


class GPR4:
    def __init__(self, parameters, measurements, sigma):
        self.x = parameters
        self.y = np.array(measurements)
        self.sigma = sigma
        self.mean_y = self.y.mean()
        if len(self.y) > 1:
            self.std_y = self.y.std()
        else:
            self.std_y = 1
        self.y -= self.mean_y

    def kernel(self, x1, x2):
        if isinstance(x1, (int, float)):
            distance_squared = ((x1-x2)**2)
        else:
            distance_squared = ((x1-x2)**2).sum()
        return np.exp(-distance_squared/(2*self.sigma**2))
    
    def estimate(self, query_parameter):        
        kernels_x_query = np.array([self.kernel(x, query_parameter)
                                    for x in self.x])
        kernels_x_x = np.array([[
            self.kernel(x1, x2) for x1 in self.x]
            for x2 in self.x])
        weights= kernels_x_query.T @ np.linalg.inv(kernels_x_x)
        expectation = self.mean_y + weights @ self.y
        uncertainty_squared = 1 - weights @ kernels_x_query
        uncertainty = np.sqrt(uncertainty_squared)
        return expectation, self.std_y*uncertainty
    

def plot_gpr(x, expectation, uncertainty, values=None, title=None):
    matplotlib.rcParams.update({'font.size': 24})
    fig, ax  = plt.subplots(figsize=(11,9))
    plt.rc('figure', titlesize=30)  
    #in
    if values is not None:
        ax.plot(values[:,0], values[:,1], 'o', color='tab:brown', ms=20)
    ax.plot(x, expectation, '-.', linewidth=4.0)
    ax.fill_between(x, expectation-uncertainty, expectation+uncertainty, alpha=0.2)
    if title is not None:
        ax.set_title(title, )
    ax.set_xlabel("Parameters Values")
    ax.set_ylabel("CPU Time")
    plt.show()
    plt.close()


def init_plot(uncertainty=.8, ):
    x = np.linspace(0,1,100)
    y = np.ones((100,)) * cpu
    plot_gpr(x, y, uncertainty, values=None, title="Initial System")



#gpr = GPR4(.5, 1.2, 8)
def init_point(sigma=0.15, ):
    gpr = GPR4([0.50, 0.0], [1.52, 1.21], sigma)
    x_hats= np.linspace(0,1,100)
    y_hats, sigma_y_hats = zip(*[gpr.estimate(x_hat) for x_hat in x_hats])
    plot_gpr(x_hats, y_hats, np.array(sigma_y_hats), np.array([[0.5, 1.52],
                                                     [0, 1.21]]),
                                                     title="Two points")

def init_pointb(sigma=0.15, ):
    gpr = GPR4([0.50], [1.52,], sigma)
    x_hats= np.linspace(0,1,100)
    y_hats, sigma_y_hats = zip(*[gpr.estimate(x_hat) for x_hat in x_hats])
    plot_gpr(x_hats, y_hats, np.array(sigma_y_hats), np.array([[0.5, 1.51],
                                                    ]), title="Baseline")


def init_point3(sigma=0.15, title="Three Points"):
    x = [0.5, 0.0, 0.17]
    y = [1.52, 1.21, 1.31]
    gpr = GPR4(x, y, sigma)
    x_hats= np.linspace(0,1,100)
    y_hats, sigma_y_hats = zip(*[gpr.estimate(x_hat) for x_hat in x_hats])
    plot_gpr(x_hats, y_hats, np.array(sigma_y_hats), np.array([x,y]).T,
                                                     title=title)

def lowest_point(sigma=0.15,  title="Design Next Experiment"):
    x = [0.5, 0.0, ]
    y = [1.52, 1.21]
    x = [0.5, .0, 0.17]
    y = [1.52, 1.21, 1.31]
    values = np.array(list(zip(x,y)))
    matplotlib.rcParams.update({'font.size': 24})
    gpr = GPR4(x, y, sigma)
    x_hats= np.linspace(0,1,100)
    y_hats, uncertainty = zip(*[gpr.estimate(x_hat) for x_hat in x_hats])
    uncertainty = np.array(uncertainty)
    fig, ax  = plt.subplots(figsize=(11,9))
    if values is not None:
        ax.plot(values[:,0], values[:,1], 'o', color='tab:brown')
    ax.plot(x_hats, y_hats, '-.')
    ax.fill_between(x_hats, y_hats-uncertainty, y_hats+uncertainty, alpha=0.2)
    lowerbound = y_hats-uncertainty
    ax.plot(x_hats, lowerbound, color='red')
    min_y = lowerbound.min()
    indexes = np.where(lowerbound==min_y)[0][0]
    
    min_x = x_hats[np.where(lowerbound==min_y)]
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Parameters Values")
    ax.set_ylabel("CPU Time")
    ax.annotate('Best Possible Out Come', 
                xy=(min_x, min_y), xytext=(min_x+.1, min_y-.01),
                arrowprops=dict(facecolor='green', shrink=0.04))
    plt.show()
    plt.close()


#init_plot()
#init_pointb()
#init_point3()
#lowest_point()

def bo_run():
    bo= BayesianOptimizer(num_parameters=7)
    x = []
    y = []
    for _ in range(45):
        parameter = bo.ask()
        cpu_time = jit_plus_server(parameter)
        bo.tell(parameter, cpu_time)
        print(_,  parameter, round(cpu_time, 2), )
        
        x.append(_)

    bo.plot(np.array(x),bo.measurements)

bo_run()
