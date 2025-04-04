import neat
import robotica
import pickle
import os
import cv2
import sys

Checkpoint_Dir = "checkpoints"

# Modify the following variables to fine tune the code.
Number_Generations = 10
max_Training_Time = 40 # 20 steps equal one second

def count_files(directory):
    try:
        return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
    except FileNotFoundError:
        print("Directory not found.")
        return -1

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

    total_fitness = 0
    prev_readings = None

    prev_front_sonar = robot.get_sonar()[3]  # Front sonar
    prev_left_sonar = robot.get_sonar()[1]   # Left side sonar
    prev_right_sonar = robot.get_sonar()[5]  # Right side sonar
    prev_rear_sonar = robot.get_sonar()[7]   # Rear sonar

    stuck_steps = 0
    line_lost_steps = 0
    
    time_step= 0

    while coppelia.is_running() and time_step < max_Training_Time:
        readings = robot.get_sonar()
        img = robot.get_image()
        line_detected, cx = process_camera_image(img)
        readings = [0.5 * new + 0.5 * old for new, old in zip(readings, prev_readings)] if prev_readings else readings
        prev_readings = readings
        
        image_center = img.shape[1] // 2 if img is not None else 0
        alignment_factor = ((cx - image_center) / image_center) if line_detected else 0
        
        lspeed = 0.5 - alignment_factor
        rspeed = 0.5 + alignment_factor
        
        outputs = net.activate(readings + [r / 1.0 for r in robot.get_lidar()[:32]])
        
        lspeed += outputs[0]
        rspeed += outputs[1]
        
        robot.set_speed(lspeed, rspeed)
        
        front_sonar = readings[3]  # Front sonar
        left_sonar = readings[1]   # Left side sonar
        right_sonar = readings[5]  # Right side sonar
        rear_sonar = readings[7]   # Rear sonar

        if (abs(front_sonar - prev_front_sonar) < 0.01 and 
            abs(rear_sonar - prev_rear_sonar) < 0.01 and 
            abs(left_sonar - prev_left_sonar) < 0.01 and 
            abs(right_sonar - prev_right_sonar) < 0.01):
            stuck_steps += 1
        else:
            stuck_steps = 0

        if not line_detected:
            line_lost_steps += 1
        else:
            line_lost_steps = 0 

        prev_front_sonar = front_sonar
        prev_left_sonar = left_sonar
        prev_right_sonar = right_sonar
        prev_rear_sonar = rear_sonar

        fitness = calculate_fitness(line_detected, alignment_factor, stuck_steps, line_lost_steps, readings, lspeed, rspeed)
        total_fitness += fitness
        time_step += 1
    coppelia.stop_simulation()
    return total_fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")

def calculate_fitness(line_detected, alignment_factor, stuck_steps, line_lost_steps, readings, lspeed, rspeed):
    fitness = 0
    avg_speed = (lspeed + rspeed) / 2
    turn_amount = abs(lspeed - rspeed)

    # === 1. Following the line well ===
    if line_detected:
        alignment_score = 1 - abs(alignment_factor)
        fitness += 20 * alignment_score
        if abs(alignment_factor) < 0.1: # Keeps the robot centered
            fitness += 5

        if avg_speed > 0:  # Forward motion when tracking the line
            fitness += 5 
    else:
        fitness -= 5 * (line_lost_steps / 20) 

    # === 2. Obstacle avoidance ===
    if readings[3] < 0.1 or readings[4] < 0.2:  # Front too close
        fitness *= 0.4
    elif readings[1] < 0.1 or readings[5] < 0.4:  # Sides too close
        fitness *= 0.6
    elif readings[6] < 0.1 or readings[7] < 0.2:  # Back too close
        fitness *= 0.6
    else:
        fitness += 3  # Clean path bonus

    # === 3. Stuck penalty ===
    if stuck_steps > 10:
        fitness *= 0.5

    # === 4. Movement style penalties ===
    if turn_amount > 1.0: # Goes in circles
        fitness *= 0.7
        
    if line_detected and abs(alignment_factor) < 0.2 and avg_speed > 0.2: #Smooth forward tracking
        fitness += 3 
        
    if avg_speed < 0: # Backwards movement
        fitness *= 0.5  
    elif avg_speed < 0.1: # Barely moving forward
        fitness *= 0.7 
    else: # Moving forward decently
        fitness += 2


    return fitness

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    num_files = count_files(Checkpoint_Dir)

    if num_files > 0:
        latest_checkpoint = os.path.join(Checkpoint_Dir, f"neat_checkpoint-{num_files - 1}")
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print("Starting a fresh training session.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(1, filename_prefix=f"{Checkpoint_Dir}/neat_checkpoint-"))

    winner = pop.run(eval_genomes, Number_Generations)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Training completed! Best genome saved.")

def main():
    config_path = "neat_config.txt"
    numFiles = count_files(Checkpoint_Dir)
    neat_output_path = "output/neat_output.txt"

    # Step 1: Check for best_genome.pkl
    if os.path.exists("best_genome.pkl"):
        answer = input("A trained model was found. Do you want to test it? (yes/no): ").strip().lower()
        if answer == "yes":
            with open("best_genome.pkl", "rb") as f:
                best_genome = pickle.load(f)

            print("Testing best genome...")
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
            return  # End program after testing
        
        else:
            answer2 = input("Do you want to delete the best_genome? (yes/no): ").strip().lower()
            
            if answer2 == "yes":
                os.remove("best_genome.pkl")


    # Step 2: Check for checkpoint files
    if numFiles == 0:
        print("No checkpoints found. Starting new training session.")
        run_neat(config_path)
        return

    # Step 3: Ask if user wants to continue training from last checkpoint
    print("Checkpoint files detected.")
    answer = input("Do you want to continue training from the last checkpoint? (yes/no): ").strip().lower()
    if answer == "yes":
        run_neat(config_path)
    else:
        # Delete all checkpoint files and neat_output file
        print("Deleting existing checkpoints and starting fresh training.")
        if os.path.exists(Checkpoint_Dir):
            for file in os.listdir(Checkpoint_Dir):
                file_path = os.path.join(Checkpoint_Dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        if os.path.exists(neat_output_path):
            try:
                os.remove(neat_output_path)
            except Exception as e:
                print(f"Error deleting neat_output.txt: {e}")

        run_neat(config_path)

if __name__ == '__main__':
    main()
