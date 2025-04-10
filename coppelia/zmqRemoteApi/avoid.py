import neat
import robotica
import pickle
import os
import cv2
import numpy as np

Checkpoint_Dir = "checkpoints"

# Modify the following variables to fine tune the training.
Number_Generations = 30
max_Training_Time = 600 # 20 steps equal one second

def count_files(directory):
    try:
        return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
    except FileNotFoundError:
        print("Directory not found.")
        return -1

def process_camera_image(img):
    # Convert BGR image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV (red spans across low and high hue values)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _, img_width = mask.shape
    line_detected = False
    cx = None

    for contour in contours:
        _, _, w, _ = cv2.boundingRect(contour)
        if w >= img_width // 5:  # Ensure the line is large enough
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

    line_lost_steps = 0
    backwards_steps = 0
    turn_steps = 0
    stuck_steps = 0
    time_step= 0

    total_fitness = 0
    
    while coppelia.is_running() and time_step < max_Training_Time:
        readings = robot.get_sonar()
        img = robot.get_image()
        line_detected, cx = process_camera_image(img)
        
        image_center = img.shape[1] // 2 if img is not None else 0
        alignment_factor = ((cx - image_center) / image_center) if line_detected else 0
        
        outputs = net.activate(readings)
        lspeed = outputs[0]
        rspeed = outputs[1]
        robot.set_speed(lspeed, rspeed)
        
        avg_speed = (lspeed + rspeed) / 2
        turn_amount = abs(lspeed - rspeed)
        
        if not line_detected:
            line_lost_steps += 1
        else:
            line_lost_steps = 0
            
        if(avg_speed < 0):
            backwards_steps += 1
        else:
            backwards_steps = 0
            
        if any(distance < 0.1 for distance in readings):
            stuck_steps += 1
        else:
            stuck_steps = 0
            
        if(turn_amount > 1.5):
            turn_steps += 1
        else:
            turn_steps = 0
        
        if(turn_steps > 60 or backwards_steps > 60 or stuck_steps > 60): # Turns in circles, goes backwards or gets stuck for more than 3 seconds
            break

        fitness = calculate_fitness(line_detected, alignment_factor, line_lost_steps, readings, avg_speed, turn_amount)
        total_fitness += fitness
        time_step += 1
        
    coppelia.stop_simulation()
    return total_fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness} \n")
        
def calculate_fitness(line_detected, alignment_factor, line_lost_steps, readings, avg_speed, turn_amount):
    fitness = 0

    # === Line detection and following rewards / penalties ===
    if line_detected:
        abs_alignment_factor = abs(alignment_factor)

        # Reward: Strongly reward being near the center
        fitness += 60 * (1 - abs_alignment_factor)

        # Bonus for being very close to center
        if abs_alignment_factor < 0.1:
            fitness += 40 * (1 - abs_alignment_factor)
        
        # Bonus for having a positive speed when detecting the line
        if(abs(avg_speed)) > 0.5 and turn_amount < 1.5:
            fitness * 10 * abs(avg_speed)

    else:
        # No reward if line not detected
        abs_alignment_factor = 1.0

    # Penalize time spent off the line
    if line_lost_steps > 0:
        fitness -= line_lost_steps * 5

    if line_detected and abs_alignment_factor > 0.7:
        fitness -= abs_alignment_factor * 30

    # === Obstacle avoidance penalty ===
    if any(distance < 0.4 for distance in readings):
        fitness -= 15 * abs(avg_speed) 
        
    # === Movement penalties ===
    if turn_amount > 1.5:
        fitness -= 10 * turn_amount  # Penalty for spinning too much

    if avg_speed < 0:  # Moving backwards
        fitness -= 10 * abs(avg_speed)    
        
          
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
                outputs = best_net.activate(readings)
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
