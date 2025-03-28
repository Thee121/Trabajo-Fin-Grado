import numpy as np 
import neat
from neat.checkpoint import Checkpointer
import robotica
import pickle
import glob
import os
import cv2


CHECKPOINT_DIR = "checkpoints"

def load_previous():
    """Loads the latest NEAT checkpoint from the folder."""
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/neat_checkpoint-*.pkl")
    
    if not checkpoints:
        return None  # No checkpoints found

    # Extract generation numbers and find the latest one
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('-')[-1].split('.')[0]))
    
    print(f"Resuming from latest checkpoint: {latest_checkpoint}")
    return Checkpointer.restore_checkpoint(latest_checkpoint)

def save_file(population, generation):
    """Saves the NEAT checkpoint in a dedicated folder."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)  # Create folder if not exists

    checkpoint = Checkpointer(generation_interval=1, filename_prefix=f"{CHECKPOINT_DIR}/neat_checkpoint-")
    checkpoint.save_checkpoint(population, population.species, population.reproduction, generation)

    print(f"Checkpoint saved for generation {generation}")

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
    
    line_detected = np.count_nonzero(edges) > 500
    
    if line_detected:
        largest_contour = max(contours, key=cv2.contourArea, default=None)
        if largest_contour is not None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                return True, cx  # Return centroid x position
    
    return False, None

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    
    total_reward = 0
    max_time_steps = 200
    time_step = 0
    prev_readings = None
    stuck_steps = 0
    prev_ls, prev_rs = 0, 0
    
    while coppelia.is_running() and time_step < max_time_steps:
        readings = normalize_readings(robot.get_sonar())
        lidar_data = normalize_readings(robot.get_lidar()[:32])
        img = robot.get_image()
        cv2.imshow('Camera Feed', img)
        cv2.waitKey(1)
        
        line_detected, _ = process_camera_image(img)
        
        readings = filter_sonar(readings, prev_readings)
        prev_readings = readings
        inputs = readings + lidar_data
        outputs = net.activate(inputs)
        
        base_speed = 2.0
        adjusted_speed = adjust_speed(base_speed, readings)
        
        if line_detected:
            stuck_steps = 0
            lspeed, rspeed = 1.0 * adjusted_speed, 1.0 * adjusted_speed
        else:
            stuck_steps += 1
            lspeed, rspeed = outputs[0] * adjusted_speed, outputs[1] * adjusted_speed
        
        robot.set_speed(lspeed, rspeed)
        
        reward = get_reward(readings, line_detected, prev_ls, prev_rs, stuck_steps)
        total_reward += reward
        
        prev_ls, prev_rs = lspeed, rspeed
        time_step += 1
    
    coppelia.stop_simulation()
    return total_reward

def line_search(robot):
    for _ in range(10):  # Rotate to scan surroundings
        robot.set_speed(0.5, -0.5)
        img = robot.get_image()
        line_detected, cx = process_camera_image(img)
        if line_detected:
            return 0.8, 0.8  # Move forward when line is found
    return 0.5, -0.5  # Default rotation if no line is found


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def normalize_readings(readings, min_val=0, max_val=1):
    return [(r - min_val) / (max_val - min_val) for r in readings]

def get_reward(readings, line_status, prev_ls, prev_rs, stuck_steps):
    reward = 0
    
    # Strong reward for following the line
    if line_status:
        reward += 10
    
    # Penalty for obstacles too close
    if readings[3] < 0.1 or readings[4] < 0.2:
        reward -= 5
    elif readings[1] < 0.1 or readings[5] < 0.4:
        reward -= 2
    
    # Penalty for excessive spinning
    if abs(prev_ls - prev_rs) > 1.5:
        reward -= 3  
    
    # Penalty if stuck (not moving forward for multiple steps)
    if stuck_steps > 10:
        reward -= 5
    
    return reward

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    population = neat.Population(config)
    print("Starting training session")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))
    
    winner = population.run(eval_genomes, 5)
            
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
            
    print("Training completed! Best genome saved.")

def main():
    config_path = "neat_config.txt"
    if os.path.exists("best_genome.pkl"):
        print("Best genome found, skipping training")
    else:
        print("No trained model found. Training a new one")
        run_neat(config_path)
    
    # Load best genome and test it
    if os.path.exists("best_genome.pkl"):
        with open("best_genome.pkl", "rb") as f:
            best_genome = pickle.load(f)
    
    print("Testing best trained genome")
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