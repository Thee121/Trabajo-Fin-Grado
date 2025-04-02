import neat
import robotica
import pickle
import os
import cv2
import sys


Checkpoint_Dir = "checkpoints"
Max_Time_Steps  = 500
Number_Generations = 20; 

def count_files(directory):
    try:
        return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
    except FileNotFoundError:
        print("Directory not found.")
        return -1
    
def load_latest_checkpoint():
    checkpoint_files = [f for f in os.listdir(Checkpoint_Dir) if f.startswith("neat_checkpoint-")]
    if not checkpoint_files:
        return None, 0  # No checkpoints available
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('-')[-1]))
    generation = int(latest_checkpoint.split('-')[-1])
    return os.path.join(Checkpoint_Dir, latest_checkpoint), generation

def validate_checkpoint_settings(config, population):
    numGenomes = 0
         
    if config.pop_size != population.config.pop_size:  # Example check, adjust based on need
        print(f"ERROR: Population size mismatch! Checkpoint: {numGenomes}, Current: {config.pop_size}")
        sys.exit(1)

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
    time_step = 0
    prev_readings = None
    stuck_steps = 0
    line_lost_steps = 0
    
    while coppelia.is_running() and time_step < Max_Time_Steps:
        readings = robot.get_sonar()
        img = robot.get_image()
        
        line_detected, cx = process_camera_image(img)
        readings = [0.5 * new + 0.5 * old for new, old in zip(readings, prev_readings)] if prev_readings else readings
        prev_readings = readings
        
        image_center = img.shape[1] // 2 if img is not None else 0
        alignment_factor = ((cx - image_center) / image_center) if line_detected else 0
        
        # The robot will move straight when aligned with the line, and turn when misaligned
        lspeed = 0.5 - alignment_factor
        rspeed = 0.5 + alignment_factor
        
        # Apply NEAT neural network output for fine-tuning
        outputs = net.activate(readings + [r / 1.0 for r in robot.get_lidar()[:32]])
        
        lspeed += outputs[0]
        rspeed += outputs[1]
        
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
    reward = 0
    
    if line_detected:
        reward += 12 * (1 - abs(alignment_factor))
        # Extra reward for good alignment
        if abs(alignment_factor) < 0.1:  
            reward += 3
    # Apply a penalty that scales with the time lost
    else:
        reward -= 2 * (line_lost_steps / 20)
    
    # Reduce reward for critical obstacles in front
    if readings[3] < 0.1 or readings[4] < 0.2:
        reward *= 0.5
    # Less severe obstacle avoidance
    elif readings[1] < 0.1 or readings[5] < 0.4:
        reward *= 0.7  
    # Small reward for clear path
    else:
        reward += 2  
    
    if stuck_steps > 10:
        reward *= 0.5
        
    # Time robot stays on track
    if line_detected:
        reward += 5 * (1 - (line_lost_steps / max(float('inf'), stuck_steps)))  
    else:
        reward -= 1
    
    # Penalize excessive turning
    if abs(lspeed - rspeed) > 1.5:
        reward *= 0.8  
    
    return reward

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    latest_checkpoint, last_generation = load_latest_checkpoint()
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint} (Generation {last_generation + 1})")
        pop = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
        validate_checkpoint_settings(config, pop)
    else:
        pop = neat.Population(config)
        last_generation = 0
        print("Starting fresh NEAT training session")
    
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(1, filename_prefix=f"{Checkpoint_Dir}/neat_checkpoint-"))
    
    log_file = open("neat_output.txt", "a")
    sys.stdout = log_file
    
    generations_to_run = Number_Generations - last_generation
    winner = pop.run(eval_genomes, generations_to_run)
    
    sys.stdout = sys.__stdout__
    log_file.close()
    
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Training completed! Best genome saved.")

def main():
    config_path = "neat_config.txt"
    numFiles = count_files(Checkpoint_Dir)
    
    
    if not os.path.exists("best_genome.pkl"):
        print("No trained model found.")
        if(numFiles != 0):
            print("There are checkpoint files present")
            if(Number_Generations == numFiles):
                print("All generations have been run. Cleaning checkpoint directory to train a fresh best genome")
                if os.path.exists(Checkpoint_Dir):
                    for file in os.listdir(Checkpoint_Dir):
                        file_path = os.path.join(Checkpoint_Dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")

                    print("Checkpoint directory cleaned. Starting new training session.")
    
        run_neat(config_path)
    
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
            robot.set_speed(outputs[0], outputs[1])

        coppelia.stop_simulation()
        print("Best genome test completed!")

if __name__ == '__main__':
    main()
