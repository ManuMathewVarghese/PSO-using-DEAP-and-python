import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools
from deap import creator
from mpl_toolkits.mplot3d import Axes3D

# Particle Generator
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

#To update velocity and position of particles
def updateParticle(part, best,w1, c1, c2):
    w1_array = (random.uniform(0, w1) for _ in range(len(part))) #w1 coeff
    c1_array = (random.uniform(0, c1) for _ in range(len(part))) #c1 coeff
    c2_array = (random.uniform(0, c2) for _ in range(len(part))) #c2 coeff

    # calculating velocity term = inertia * current speed
    inertia_term = map(operator.mul, w1_array,part.speed)

    # cognitive term = c1 * (particle best - particle current)
    cognitive_term = map(operator.mul, c1_array, map(operator.sub, part.best, part))

    # social term = c1 * (pop best - pop current)
    social_term = map(operator.mul, c2_array, map(operator.sub, best, part))

    #velocity update
    part.speed = list(map(operator.add, inertia_term, map(operator.add, cognitive_term, social_term)))
    part[:] = list(map(operator.add, part, part.speed))

#Test function
def test_function(args):
    return (args[0]**2+args[1]-11)**2 + (args[0] + args[1]**2-7)**2, # Himmelblau's function

def fitness(particle):
    return test_function(particle)

#To plot performance graph
def plot_fit(best,gen_max):
    plt.plot(range(0,gen_max+1,10), best)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()

#Containers
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,smin=None, smax=None, best=None)
#Toolbox
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
from deap import benchmarks
toolbox.register("evaluate", fitness)
toolbox.register("update", updateParticle)

#To plot landscape and particles
def pso_plot(pop):
    fig = plt.figure()
    ax = Axes3D(fig, azim=-29, elev=49)
    X = Y = np.arange(-7, 7, 0.1)
    X, Y = np.meshgrid(X, Y)
    out = [test_function([x,y])[0] for x,y in zip(X.flat, Y.flat)]
    Z = np.fromiter(out, dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
    pop = np.array(pop)
    out_ = [test_function([x,y])[0] for x,y in zip(pop[:,0],pop[:,1])]
    ax.scatter(pop[:, 0], pop[:, 1], out_, c="red",s=100,alpha=1)
    ax.plot_surface(X, Y, Z,cmap="Blues",alpha=0.8)
    plt.show()


def main():
    #Initialize population
    pop = toolbox.population(n=100)
    gen_max = 1000
    best = None
    best_fit = []
    w1 = 0.72984
    c1 = c2 = 1.496180
    for i in range(gen_max+1):

        ####modifications#######( Comment entire modification for config 1)
        '''
        Over generations velocity term and cognitive term is reduced to increase local exploration
        while social term is reduced to decrease global exploration.
        
        This is done based on the intuition that over generation all the particles will be more closer 
        towards global optima and should increase local search to converge to global optima. 
        '''
        #velocity damping using inertia over generations
        w1 = 0.72984 - (i/(gen_max*2))

        # Adjusting coeff to increase local exploration and decrease global exploration over generations
        c1 = 1.496180 + (i/gen_max)
        c2 = 1.496180 - (i / gen_max)
        ####################################################


        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            # Update P_best
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            # Update G_best
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        # Update velocity and position
        for part in pop:
            toolbox.update(part, best,w1,c1, c2)
        if i % 10 == 0:
            best_fit.append(best.fitness.values)
            print("Generation:{}   Fitness:{}".format(i,best_fit[-1]))
            ###uncomment to see 3d plot of particles moving around landscape
            #pso_plot(pop)
    ##uncomment to see best value plot
    #plot_fit(best_fit,gen_max)

if __name__ == "__main__":
    main()





