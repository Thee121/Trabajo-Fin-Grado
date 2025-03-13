import numpy as np
import cv2
import neat
import robotica
import pickle
import os

# Load previous progress if available
def load_checkpoint():
    if os.path.exists("neat_checkpoint.pkl"):
        with open("neat_checkpoint.pkl", "rb") as f:
            return pickle.load(f)
    return None

# Save progress for later use
def save_checkpoint(population, generation):
    checkpoint_data = {
        "population": population,
        "generation": generation
    }
    with open("neat_checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint_data, f)
    print("Checkpoint saved! Training can be resumed later.")

# Configuration for NEAT
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    coppelia.start_simulation()
    total_reward = 0
    max_time_steps = 500  # Limit training per genome
    time_step = 0

    while coppelia.is_running() and time_step < max_time_steps:
        readings = robot.get_sonar()
        inputs = [r for r in readings]
        outputs = net.activate(inputs)
        lspeed, rspeed = outputs[0] * 2, outputs[1] * 2
        robot.set_speed(lspeed, rspeed)
        
        reward = get_reward(readings)
        total_reward += reward
        time_step += 1
    
    coppelia.stop_simulation()
    print(f"Genome evaluated with total reward: {total_reward}")
    return total_reward


def get_reward(readings):
    """Reward function based on sensor readings."""
    if readings[3] < 0.1 or readings[4] < 0.2:
        return -10  # Penalize collisions
    elif readings[1] < 0.1 or readings[5] < 0.4:
        return -5  # Slight penalty for obstacles nearby
    return +1  # Reward for moving freely


def run_neat(config_path, num_generations):
    """Run NEAT algorithm with given configuration."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    checkpoint = load_checkpoint()
    if checkpoint:
        population = checkpoint["population"]
        generation = checkpoint["generation"]
        print(f"Resuming training from generation {generation}")
    else:
        population = neat.Population(config)
        generation = 0
        print("Starting new training session...")
    
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    def eval_genomes(genomes, config):
        """Evaluate multiple genomes."""
        nonlocal generation
        if generation >= num_generations:
            return  # Stop if we have reached the max generations
        
        print(f"\n****** Running generation {generation} ******\n")
        for genome_id, genome in genomes:
            genome.fitness = eval_genome(genome, config)
        
        generation += 1
        save_checkpoint(population, generation)
    
    while generation < num_generations:
        population.run(eval_genomes, 1)  # Run only one generation at a time
    
    winner = max(population.population.values(), key=lambda g: g.fitness)
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Training completed! Best genome saved.")


def main():
    config_path = "neat_config.txt"  # Ensure you have a NEAT config file
    num_generations = 2  # Set the number of generations dynamically if needed
    run_neat(config_path, num_generations)
    
    # Load best genome and test it
    if os.path.exists("best_genome.pkl"):
        with open("best_genome.pkl", "rb") as f:
            best_genome = pickle.load(f)
    else:
        print("No trained model found. Training a new one...")
        run_neat(config_path, num_generations)
        with open("best_genome.pkl", "rb") as f:
            best_genome = pickle.load(f)
    
    print("Testing best trained genome...")
    best_net = neat.nn.FeedForwardNetwork.create(best_genome, neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    ))
    
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    coppelia.start_simulation()
    
    while coppelia.is_running():
        readings = robot.get_sonar()
        inputs = [r for r in readings]
        outputs = best_net.activate(inputs)
        robot.set_speed(outputs[0] * 2, outputs[1] * 2)
    
    coppelia.stop_simulation()
    print("Best genome test completed!")

if __name__ == '__main__':
    main()
