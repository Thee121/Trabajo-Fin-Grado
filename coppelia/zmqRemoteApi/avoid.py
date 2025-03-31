import numpy as np 
import neat
import robotica
import pickle
import os
import cv2
import sys
import shutil
        
def avoidObstacle(readings, base_speed=1.5):
    if (readings[3] < 0.1) or (readings[4] < 0.2):
        lspeed, rspeed = adjust_speed(base_speed * 0.15, readings), adjust_speed(base_speed * -0.8, readings)
    elif readings[1] < 0.1:
        lspeed, rspeed = adjust_speed(base_speed * 1.5, readings), adjust_speed(base_speed * 0.6, readings)
    elif readings[5] < 0.4:
        lspeed, rspeed = adjust_speed(base_speed * 0.15, readings), adjust_speed(base_speed * 0.9, readings)
    else:
        lspeed, rspeed = adjust_speed(base_speed, readings), adjust_speed(base_speed, readings)
    return lspeed, rspeed

def line_search(robot):
    for _ in range(10):  # Rotate to scan surroundings
        robot.set_speed(0.5, -0.5)
        img = robot.get_image()
        line_detected, cx = process_camera_image(img)
        print(line_detected)
        if line_detected:
            return 0.8, 0.8  # Move forward when line is found
    return 0.5, -0.5  # Default rotation if no line is found

def filter_sonar(readings, prev_readings, alpha=0.5):
    if prev_readings is None:
        return readings
    return [alpha * new + (1 - alpha) * old for new, old in zip(readings, prev_readings)]

def adjust_speed(base_speed, readings, min_dist=0.2, max_dist=1.0):
    min_reading = min(readings)
    if min_reading < min_dist:
        return base_speed * 0.5 
    elif min_reading > max_dist:
        return base_speed * 1.2
    return base_speed

def process_camera_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 100, cv2.THRESH_BINARY_INV)  # Detect black lines
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    _, img_width = gray.shape
    
    line_detected = False
    cx = None
    
    for contour in contours:
        _, _, w, _ = cv2.boundingRect(contour)
        if w >= img_width // 5:  # Ensure the line is on the floor and fully in view
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                line_detected = True
                break
    
    return line_detected, cx


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    
    min_speed = 2.0
    
    coppelia.start_simulation()
    
    total_reward = 0
    max_time_steps = 200
    time_step = 0
    prev_readings = None
    stuck_steps = 0
    prev_ls, prev_rs = 0, 0
    line_lost_steps = 0
    
    while coppelia.is_running() and time_step < max_time_steps:
        readings = robot.get_sonar()
        img = robot.get_image()
        cv2.imshow('Camera Feed', img)
        cv2.waitKey(1)
        
        line_detected, cx = process_camera_image(img)
        readings = filter_sonar(readings, prev_readings)
        prev_readings = readings
        
        inputs = readings + normalize_readings(robot.get_lidar()[:32])
        outputs = net.activate(inputs)
        
        if line_detected:
            stuck_steps = 0
            line_lost_steps = 0
            
            image_center = img.shape[1] // 2  # Horizontal center of the image
            alignment_factor = (cx - image_center) / image_center  # Normalized alignment factor (-1 to 1)
            turn_factor = alignment_factor * 0.75  # Adjust turning based on alignment
            
            lspeed, rspeed = min_speed, min_speed
            lspeed += turn_factor
            rspeed -= turn_factor
        else:
            stuck_steps += 1
            line_lost_steps += 1
            
            if line_detected:  # Prioritize searching as soon as the line is detected
                base_ls, base_rs = line_search(robot)
            else:
                base_speed = min_speed if line_detected else outputs[3] * 1.5  # Dynamic base speed
                base_ls, base_rs = avoidObstacle(readings, base_speed)

            turn_factor = outputs[2] * 0.75
            speed_factor = outputs[3] * 1.25
            
            lspeed = (base_ls + turn_factor) * (1 + speed_factor)
            rspeed = (base_rs - turn_factor) * (1 + speed_factor)
        
        robot.set_speed(lspeed, rspeed)
        
        reward = get_reward(readings, line_detected, line_lost_steps, stuck_steps)
        

        
        total_reward += reward
        
        prev_ls, prev_rs = lspeed, rspeed
        time_step += 1
    
    coppelia.stop_simulation()
    return total_reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def normalize_readings(readings, min_val=0, max_val=1):
    return [(r - min_val) / (max_val - min_val) for r in readings]

def get_reward(readings, line_detected, line_lost_steps, stuck_steps):
    reward = 0
    
    # Follows the line
    if line_detected:
        reward += 12
    
    # Obstacles too close
    if readings[3] < 0.1 or readings[4] < 0.2:
        reward -= 4
    elif readings[1] < 0.1 or readings[5] < 0.4:
        reward -= 3
    
    # Stuck for too long
    if stuck_steps > 10:
        reward -= 3
    
    # Lost the line for too long
    if line_lost_steps > 15:
        reward -= 5 

    return reward

def run_neat(config_path):
    
    log_file = open("neat_output.txt", "a")
    sys.stdout = log_file
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    population = neat.Population(config)
    print("Starting training session")
    
    CHECKPOINT_DIR = "checkpoints"
    stats = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1, filename_prefix=f"{CHECKPOINT_DIR}/neat_checkpoint-"))    
    
    winner = population.run(eval_genomes, 10) # Runs up to X generations
    
    sys.stdout = sys.__stdout__
    log_file.close()
    
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
            
    print("Training completed! Best genome saved.")

def main():
    config_path = "neat_config.txt"
    checkpoint_dir = "checkpoints"

    if os.path.exists("best_genome.pkl"):
        choice = input("Best genome found. Do you want to continue training from the last checkpoint? (y/n): ").strip().lower()
        if choice == "y":
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("neat_checkpoint-")]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split("-")[-1]))  # Get the latest checkpoint
                print(f"Resuming training from {latest_checkpoint}...")

                # Redirect output to log file
                log_file = open("neat_output.txt", "a")
                sys.stdout = log_file

                # Load the checkpoint and resume training
                population = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint))
                population.run(eval_genomes, 10)  # Continue training
                
                # Restore stdout
                sys.stdout = sys.__stdout__
                log_file.close()

                # Append cleaned logs to neat_output_clean.txt
                try:
                    with open("neat_output.txt", "r") as fr:
                        lines = fr.readlines()

                    with open("neat_output_clean.txt", "a") as fw:  # Append instead of overwrite
                        for line in lines:
                            if line.strip('\n') not in ['*** connecting to coppeliasim', '*** getting handles PioneerP3DX', '*** done']:
                                fw.write(line)
                    
                    print("neat_output file updated and cleaned!")
                except Exception as e:
                    print(f"Error while updating logs: {e}")
            else:
                print("No checkpoints found. Starting training from scratch.")
                run_neat(config_path)
        else:
            print("Testing the best trained genome.")
    else:
        print("No trained model found. Cleaning checkpoint directory...")
        
        # Delete all checkpoint files if they exist
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # In case there are subdirectories
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        print("Checkpoint directory cleaned. Starting new training session.")
        run_neat(config_path)

    # Clean unwanted log lines
    try:
        with open('neat_output.txt', 'r') as fr:
            lines = fr.readlines()

        with open('neat_output_clean.txt', 'w') as fw:
            for line in lines:
                if line.strip('\n') not in ['*** connecting to coppeliasim', '*** getting handles PioneerP3DX', '*** done']:
                    fw.write(line)
        print("neat_output file cleaned!")
    except Exception as e:
        print(f"Error while cleaning logs: {e}")

    # Load and test the best genome
    if os.path.exists("best_genome.pkl"):
        with open("best_genome.pkl", "rb") as f:
            best_genome = pickle.load(f)

        print("Testing best trained genome.")
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
            outputs = best_net.activate(readings + lidar_data[:32])
            robot.set_speed(outputs[0] * 2, outputs[1] * 2)

        coppelia.stop_simulation()
        print("Best genome test completed!")

if __name__ == '__main__':
    main()
