from queue import Full
import numpy as np 
import neat
import robotica
import pickle
import os
import cv2

def load_previous():
    if os.path.exists("neat_save.pkl"):
        with open("neat_save.pkl", "rb") as f:
            return pickle.load(f)
    return None

def save_file(population, generation):
    save_data = {
        "population": population,
        "generation": generation
    }
    with open("neat_save.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"file saved succesfully!")

def filter_sonar(readings, prev_readings, alpha=0.5):
    if prev_readings is None:
        return readings
    return [alpha * new + (1 - alpha) * old for new, old in zip(readings, prev_readings)]

def adjust_speed(base_speed, readings, min_dist=0.2, max_dist=1.0):
    min_reading = min(readings)
    if min_reading < min_dist:
        return base_speed * 0.5  # Slow down significantly when too close
    elif min_reading > max_dist:
        return base_speed * 1.2  # Slightly increase speed when obstacles are far
    return base_speed  # Default speed

def detect_line(cv_image):
    if cv_image is None:
        return False  # Ensure we have a valid image
    cv2.imshow('opencv', cv_image)
    image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height, width = edges.shape
    mid_x = width // 2
    line_present = np.sum(edges[:, mid_x - 20:mid_x + 20]) > (height * 255 * 0.05)
    
    return line_present

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    cv_image = robot.get_image()
    cv2.waitKey(1)

    
    total_reward = 0
    max_time_steps = 200  # Limit training per genome
    time_step = 0
    prev_readings = None
    line_lost_steps = 0  # Track how long the line is lost
    found_line = False  # Track if line was found after being lost

    while coppelia.is_running() and time_step < max_time_steps:
        readings = robot.get_sonar()
        lidar_data = robot.get_lidar()
        line_status = detect_line(cv_image)  # Camera-based line detection
        
        # Apply sonar filtering
        readings = filter_sonar(readings, prev_readings)
        prev_readings = readings
        
        inputs = readings + lidar_data[:32]  # Reduce lidar data size if needed
        outputs = net.activate(inputs)
        
        # Dynamic speed adjustment
        base_speed = 2.0
        adjusted_speed = adjust_speed(base_speed, readings)
        
        if line_status:
            line_lost_steps = 0  # Reset lost line counter
            if found_line:
                total_reward += 3  # Reward finding the line again
                found_line = False
        else:
            line_lost_steps += 1
            if line_lost_steps > 20:
                found_line = True  # Mark that it was lost and later found
                lspeed, rspeed = line_search_pattern(time_step)
            else:
                lspeed, rspeed = outputs[0] * adjusted_speed, outputs[1] * adjusted_speed
        
        # Define speed limits to prevent extreme turns
        max_speed = 2.0  # Maximum wheel speed
        min_speed = 0.2  # Prevent stopping completely
        
        # Normalize NEAT outputs (-1 to 1) into speed range (min_speed to max_speed)
        lspeed = min_speed + (outputs[0] + 1) / 2 * (max_speed - min_speed)
        rspeed = min_speed + (outputs[1] + 1) / 2 * (max_speed - min_speed)
        robot.set_speed(lspeed, rspeed)
        
        alpha = 0.5  # Smoothing factor (0: no change, 1: instant change)
        lspeed = alpha * lspeed + (1 - alpha) * lspeed
        rspeed = alpha * rspeed + (1 - alpha) * rspeed

        # Reward system
        reward = get_reward(readings, line_status)
        total_reward += reward
        time_step += 1
    
    coppelia.stop_simulation()
    print(f"Genome evaluated with total reward: {total_reward}")
    return total_reward

def line_search_pattern(step):
    angle = step * 0.1  # Increment angle to create a spiral motion
    speed = 1.0  # Move at a steady speed
    return speed * np.cos(angle), speed * np.sin(angle)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def get_reward(readings, line_status):
    if line_status:
        return +3  # Reward for following the line
    elif readings[3] < 0.1 or readings[4] < 0.2:
        return -5  # Penalize collisions
    elif readings[1] < 0.1 or readings[5] < 0.4:
        return -2  # Slight penalty for obstacles nearby
    return -2  # Penalize being lost

def run_neat(config_path):
    num_generations = 5 # Change for number of generations
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    saveFile = load_previous()
    if saveFile:
        population = saveFile["population"]
        generation = saveFile["generation"]
        print(f"Resuming training from generation {generation}")
    else:
        population = neat.Population(config)
        generation = 0
        print("Starting new training session...")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    while generation < num_generations:
        winner = population.run(eval_genomes, 1)
        generation += 1
        save_file(population, generation)

        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
    
    print("Training completed! Best genome saved.")

def main():
    config_path = "neat_config.txt"
    if os.path.exists("best_genome.pkl"):
        print("Best genome found, skipping training...")
    else:
        print("No trained model found. Training a new one...")
        run_neat(config_path)
    
    # Load best genome and test it
    if os.path.exists("best_genome.pkl"):
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
        lidar_data = robot.get_lidar()
        outputs = best_net.activate(readings + lidar_data[:16])
        robot.set_speed(outputs[0] * 2, outputs[1] * 2)
    
    coppelia.stop_simulation()
    print("Best genome test completed!")

if __name__ == '__main__':
    main()