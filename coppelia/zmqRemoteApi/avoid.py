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
    print(f"file saved successfully!")

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

def detect_line(robot):
    sensor_readings = robot.get_sonar()
    return any(sensor_readings)  # If any sensor detects the line, return True

def avoidObstacle(readings):
    if (readings[3] < 0.1) or (readings[4] < 0.2):
        lspeed, rspeed = +0.1, -0.8
    elif readings[1] < 0.1:
        lspeed, rspeed = +1.3, +0.6
    elif readings[5] < 0.4:
        lspeed, rspeed = +0.1, +0.9
    else:
        lspeed, rspeed = +1.5, +1.5
    return lspeed, rspeed

def process_camera_image(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_detected = np.count_nonzero(edges) > 500  # Adjust threshold as needed
    obstacle_detected = len(contours) > 0

    return line_detected, obstacle_detected

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    
    total_reward = 0
    max_time_steps = 200  # Limit training per genome
    time_step = 0
    prev_readings = None
    line_lost_steps = 0  # Track how long the line is lost
    found_line = False  # Track if line was found after being lost
    prev_ls, prev_rs = 0, 0  # Initialize previous speed values
    
    while coppelia.is_running() and time_step < max_time_steps:
        readings = robot.get_sonar()
        lidar_data = robot.get_lidar()

        # Get camera feed
        img = robot.get_image()
        cv2.imshow('Camera Feed', img)
        cv2.waitKey(1)

        # Process camera data
        line_detected, obstacle_detected = process_camera_image(img)

        # Update line_status based on camera detection
        line_status = line_detected  

        # Adjust robot behavior based on obstacle detection
        if obstacle_detected:
            total_reward -= 3  # Penalize being too close to obstacles

        readings = filter_sonar(readings, prev_readings)
        prev_readings = readings

        inputs = readings + lidar_data[:32]  
        outputs = net.activate(inputs)

        base_speed = 2.0
        adjusted_speed = adjust_speed(base_speed, readings)

        if line_status:
            line_lost_steps = 0
            if found_line:
                total_reward += 5  
                found_line = False
            if abs(outputs[0] - outputs[1]) < 0.3:  # Encourage forward movement
                lspeed, rspeed = 1.0 * adjusted_speed, 1.0 * adjusted_speed
            else:
                readings = robot.get_sonar()
                lspeed, rspeed = avoidObstacle(readings)
                robot.set_speed(lspeed, rspeed)
        else:
            line_lost_steps += 1
            if line_lost_steps > 20:
                found_line = True
                lspeed, rspeed = line_search(robot)

            else:
                lspeed, rspeed = outputs[0] * adjusted_speed, outputs[1] * adjusted_speed

        robot.set_speed(lspeed, rspeed)

        reward = get_reward(readings, line_status, prev_ls, prev_rs)
        total_reward += reward

        prev_ls, prev_rs = lspeed, rspeed
        time_step += 1
    
    coppelia.stop_simulation()
    print(f"Genome evaluated with total reward: {total_reward}")
    return total_reward

def line_search(robot):
    """Search for the line using lidar and sonar data instead of unavailable sensors."""
    sonar_readings = robot.get_sonar()
    lidar_data = robot.get_lidar()
    
    # Use lidar data to detect obstacles and possible line edges
    front_lidar = np.mean(lidar_data[:5])  # Average front lidar values
    left_lidar = np.mean(lidar_data[5:10])  # Left side
    right_lidar = np.mean(lidar_data[-10:-5])  # Right side

    # Favor forward movement if no obstacles are detected
    if front_lidar > 0.5 and max(left_lidar, right_lidar) > 0.5:
        return 1.0, 1.0  # Move forward
    elif left_lidar < right_lidar:
        return -0.2, 0.5  # Turn left slightly while moving
    else:
        return 0.5, -0.2  # Turn right slightly while moving


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def get_reward(readings, line_status, prev_ls, prev_rs):
    if line_status:
        return +5  # Stronger reward for following the line
    elif readings[3] < 0.1 or readings[4] < 0.2:
        return -5  # Penalize collisions
    elif readings[1] < 0.1 or readings[5] < 0.4:
        return -2  # Slight penalty for obstacles nearby

    # Penalize excessive spinning
    if abs(prev_ls - prev_rs) > 1.5: 
        return -3  

    return -2  # General penalty for wandering aimlessly


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
        winner = population.run(eval_genomes, 5) # Change for number of Generations
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