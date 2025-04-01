import neat
import robotica
import pickle
import os
import cv2
import sys
import shutil

def adjust_speed(speed):
    return max(min(speed, 1.5), -1.5)

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
    
    coppelia.start_simulation()
    
    total_reward = 0
    max_time_steps = 200
    time_step = 0
    prev_readings = None
    stuck_steps = 0
    line_lost_steps = 0
    
    while coppelia.is_running() and time_step < max_time_steps:
        readings = robot.get_sonar()
        img = robot.get_image()
        
        line_detected, cx = process_camera_image(img)
        readings = [0.5 * new + 0.5 * old for new, old in zip(readings, prev_readings)] if prev_readings else readings
        prev_readings = readings
        
        image_center = img.shape[1] // 2 if img is not None else 0
        alignment_factor = ((cx - image_center) / image_center) if line_detected else 0
        
        # The robot will move straight when aligned with the line, and turn when misaligned
        lspeed = 0.5 - alignment_factor * 0.2  # Adjust left speed based on alignment
        rspeed = 0.5 + alignment_factor * 0.2  # Adjust right speed based on alignment
        
        # Apply NEAT neural network output for fine-tuning
        outputs = net.activate(readings + [r / 1.0 for r in robot.get_lidar()[:32]])
        
        lspeed += outputs[0] * 0.3  # Small adjustment based on NEAT output
        rspeed += outputs[1] * 0.3  # Small adjustment based on NEAT output
        
        lspeed = adjust_speed(lspeed)
        rspeed = adjust_speed(rspeed)
        robot.set_speed(lspeed, rspeed)
        
        reward = get_reward(line_detected, alignment_factor, stuck_steps, line_lost_steps, readings, lspeed, rspeed)
        total_reward += reward
        time_step += 1
    
    coppelia.stop_simulation()
    return total_reward

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def get_reward(line_detected, alignment_factor, stuck_steps, line_lost_steps, readings, lspeed, rspeed):
    line_follow_weight = 12
    alignment_weight = 2
    obstacle_penalty_multiplier = 0.5  # Reduces reward when obstacles are too close
    stuck_penalty_multiplier = 0.7    # Reduces reward if stuck for too long
    line_lost_penalty_multiplier = 0.6  # Reduces reward if line is lost for too long
    spinning_penalty_multiplier = 0.5   # Reduces reward if spinning excessively
    
    line_follow_reward = line_follow_weight if line_detected else 0

    alignment_reward = alignment_weight if abs(alignment_factor) < 0.2 else 0

    obstacle_penalty = 1
    if readings[3] < 0.1 or readings[4] < 0.2:
        obstacle_penalty = obstacle_penalty_multiplier
    elif readings[1] < 0.1 or readings[5] < 0.4:
        obstacle_penalty = obstacle_penalty_multiplier
    
    stuck_penalty = stuck_penalty_multiplier if stuck_steps > 10 else 1

    line_lost_penalty = line_lost_penalty_multiplier if line_lost_steps > 15 else 1

    spinning_penalty = spinning_penalty_multiplier if abs(lspeed - rspeed) > 1.5 else 1

    total_reward = (
        line_follow_reward + alignment_reward
    ) * obstacle_penalty * stuck_penalty * line_lost_penalty * spinning_penalty

    return total_reward


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
                try:
                    with open("neat_output.txt", "r") as fr:
                        lines = fr.readlines()

                    with open("neat_output_clean.txt", "a") as fw:
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
