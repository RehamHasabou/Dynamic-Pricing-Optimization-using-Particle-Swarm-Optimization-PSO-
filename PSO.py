# https://colab.research.google.com/drive/1yjLrFRmG3cIPiLT8LjnOmIkT-XC5Sz1E?usp=sharing 
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
class Particle:
    def __init__(self, dim):
        self.position = np.random.rand(dim)
        self.velocity = np.random.rand(dim)
        self.fitness = float('-inf')  # Initialize with negative infinity for maximization

def objective_function(x):
    # Objective function to be maximized
    return 19606 * x[0] + 67897 * x[1] + 22632 * x[2] + 17749 * x[3]

def total_bid_constraint(x):
    # Total bid constraint function
    return 262500 - (20000 * x[0] + 75000 * x[1] + 25000 * x[2] + 20000 * x[3])

def balance_constraint(x):
    # Balance constraint function
    return x[2] - x[1]

def update_velocity(particle, particle_local_position , global_best_position,  cognitive_coeff, social_coeff):
    # Update particle velocity based on PSO equations
    new_velocity = (particle.velocity +
                    cognitive_coeff * np.random.rand(len(particle.position)) * (particle_local_position - particle.position) +
                    social_coeff * np.random.rand(len(particle.position)) * (global_best_position - particle.position))
    return new_velocity

def update_position(particle):
    # Update particle position based on velocity and bounds
    new_position = particle.position + particle.velocity
    for i in range(len(new_position)):
        new_position[i] = np.minimum(new_position[i] ,x_max[i])
        new_position[i] = np.maximum(new_position[i] ,x_min[i])
    return new_position

def PSO(objective_function, constraints, num_particles=50, max_iterations=100, cognitive_coeff=1.5, social_coeff=2.5):
    # Particle Swarm Optimization algorithm
    dim = 4
    particles = [Particle(dim) for _ in range(num_particles)]
    particle_local_fitness = float('-inf')
    global_best_fitness = float('-inf')  # Initialize with negative infinity for maximization

    best_fitness_history = []

    for iteration in range(max_iterations):
        for particle in particles:
            if all(constraint(particle.position) >= 0 for constraint in constraints) :  # Check if constraints are satisfied
                particle.fitness = objective_function(particle.position)
                if particle.fitness > particle_local_fitness:  # Check for maximization
                    particle_local_fitness = particle.fitness
                    particle_local_position = particle.position.copy()
                if particle_local_fitness > global_best_fitness:  # Check for maximization
                    global_best_fitness = particle_local_fitness
                    global_best_position = particle_local_position.copy()

        best_fitness_history.append(global_best_fitness)  # Append positive fitness value

        for particle in particles:
            particle.velocity = update_velocity(particle, particle_local_position, global_best_position, cognitive_coeff, social_coeff)
            particle.position = update_position(particle)

    return np.around(global_best_position,decimals=1), global_best_fitness, best_fitness_history



# Define constraints and bounds
constraints = [total_bid_constraint, balance_constraint]
x_max = [4 , 3 , 6 , 4]
x_min = [1 , 0.5 , 1.5 , 1]

# Run PSO algorithm
best_position, best_fitness, best_fitness_history = PSO(objective_function, constraints)
print("Optimal unit prices:")
print("Item 1:", best_position[0])
print("Item 2:", best_position[1])
print("Item 3:", best_position[2])
print("Item 4:", best_position[3])
print("Total profit:", np.around(best_fitness,decimals=2))
# Plot best fitness over iterations
plt.plot(best_fitness_history)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('PSO Optimization Progress')
plt.grid(True)
plt.show()

crisp = best_position[0]
variable_cost = np.arange(1, 4.5, 0.5)
var = ctrl.Antecedent(variable_cost, 'variable1_cost')
var['low'] = fuzz.trimf(var.universe, [1, 1 , 1.5])
var['medium'] = fuzz.trimf(var.universe, [1.5, 2, 3])
var['high'] = fuzz.trimf(var.universe, [3, 4, 4])
fuzzy_value_low_var = fuzz.interp_membership(var.universe, var['low'].mf, crisp)
fuzzy_value_medium_var = fuzz.interp_membership(var.universe, var['medium'].mf, crisp)
fuzzy_value_high_var = fuzz.interp_membership(var.universe, var['high'].mf, crisp)
print(f"Fuzzy value for 'low' at {crisp}: {fuzzy_value_low_var}")
print(f"Fuzzy value for 'medium' at {crisp}: {fuzzy_value_medium_var}")
print(f"Fuzzy value for 'high' at {crisp}: {fuzzy_value_high_var}")

simulations = 20
defuzzified_prices = []

for i in range(simulations):
    while True:
        x = np.random.uniform(0.5, 4)
        u_high = fuzz.interp_membership(var.universe, var['high'].mf, x)
        alpha = np.random.uniform(0, 1)
        if alpha <= u_high:
            defuzzified_prices.append(x)
            break

average_defuzzified_price = np.mean(defuzzified_prices)
average_defuzzified_price = np.minimum(x_max[0] , average_defuzzified_price)
average_defuzzified_price = np.maximum(x_min[0], average_defuzzified_price)
print(f"Average 'defuzzified' price  {average_defuzzified_price:.2f}")
if(262500 - (20000 * average_defuzzified_price + 75000 * best_position[1] + 25000 * best_position[2] + 20000 * best_position[3])<0):
  print("this defuzzified value violates the constraint so it's not acceptable")
else:
  print("the maximized value of the profit would be after the deffuzified price is " + str(np.around((19606 * average_defuzzified_price + 67897 *best_position[1]  + 22632 * best_position[2] + 17749 * best_position[3]),decimals=2)))
