import neat
import robotica
import pickle
import os
import cv2
import numpy as np

checkpoint_path = "output/checkpoints"
config_path = "neat_config.txt"
robot_info_path = "output/robot_info.txt"
graphs_path = "output/graphs"

Number_Generations = 120
max_Training_Time = 600 # 20 steps equal one second

def count_files(directory):
    try:
        return sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
    except FileNotFoundError:
        print("Directory not found.")
        return -1

def process_camera_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue
    lower_blue = np.array([118, 240, 240])
    upper_blue = np.array([122, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Check if line is at the bottom
    height = mask.shape[0]
    bottom_region = mask[int(height * 0.9):, :]

    on_line = cv2.countNonZero(bottom_region) > 10
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_detected = False
    cx = None

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 200:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        line_detected = True
        break
    
    image_center = img.shape[1] // 2 if img is not None else 0
    alignment_factor = ((cx - image_center) / image_center) if line_detected else 1
    return alignment_factor, on_line

def eval_genome(genome, config):    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    
    line_lost_steps = 0
    backwards_steps = 0
    turn_steps = 0
    stuck_steps = 0
    stop_steps = 0
    alignment_steps = 0
    turn_amount = 0
    time_step= 0
    
    total_fitness = 0
    
    coppelia.start_simulation()
       
    while coppelia.is_running() and time_step < max_Training_Time:
        readings = robot.get_sonar()
        img = robot.get_image()
        alignment_factor, on_line  = process_camera_image(img)
        
        outputs = net.activate(readings)
        lspeed = outputs[0]
        rspeed = outputs[1]
        robot.set_speed(lspeed, rspeed)
        
        fitness = 0
        
        avg_speed = (lspeed + rspeed) / 2
        turn_amountl = abs(lspeed - rspeed)
        turn_amountr = abs(rspeed -lspeed)
        
        if(lspeed == 0 and abs(rspeed) > 0) or (rspeed == 0 and abs(lspeed > 0)):
            turn_amount = 2
        elif(turn_amountl > turn_amountr):
            turn_amount = turn_amountl
        else:
            turn_amount = turn_amountr
            
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
        
        if(avg_speed == 0):
            stop_steps += 1
        else:
            stop_steps = 0
            
        if on_line:
            alignment_steps += 1
            line_lost_steps = 0
        
        else:
            line_lost_steps += 1
            alignment_steps = 0
            
        if(stop_steps > 100 or turn_steps > 100 or stuck_steps > 100 or backwards_steps > 100):
            break

        time_step += 1
        abs_alignment_factor = abs(alignment_factor)

        fitness = calculate_fitness(abs_alignment_factor, avg_speed, turn_amount, alignment_steps/20, backwards_steps/20, turn_steps/20, stuck_steps/20, stop_steps/20, time_step/20)
        total_fitness += fitness
        
    coppelia.stop_simulation()
    return total_fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        print(f"Genome {genome_id} fitness: {genome.fitness}")
        
def calculate_fitness(alignment_factor, avg_speed, turn_amount, alignment_steps, backwards_steps, turn_steps, stuck_steps, stop_steps, time_step):
    fitness = 0
    
    # Positive speed and no circles   
    if avg_speed > 0.1 and turn_amount < 1.5:
        
        angle_line = 1 - alignment_factor
        fitness_factor = time_step * angle_line
        
        # General movement
        fitness += fitness_factor + alignment_steps

        # How well robot is aligned
        if alignment_factor < 0.1:
            fitness += fitness_factor + alignment_steps * 3
        elif alignment_factor < 0.2:
            fitness += fitness_factor + alignment_steps * 2
        elif alignment_factor < 0.3:
            fitness += fitness_factor + alignment_steps

    # Obstacle avoidance
    if stuck_steps > 0:
        fitness -= stuck_steps
        
    # Movement penalties
    if turn_amount > 1.5: # Spinning too much
        fitness -= turn_steps 
    if avg_speed < 0:  # Moving backwards
        fitness -= backwards_steps
    if avg_speed == 0: # No movement
        fitness -= stop_steps
        
    return fitness

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    num_files = count_files(checkpoint_path)

    if num_files > 0:
        latest_checkpoint = os.path.join(checkpoint_path, f"neat_checkpoint-{num_files - 1}")
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print("Starting a fresh training session.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(1, filename_prefix=f"{checkpoint_path}/neat_checkpoint-"))

    winner = pop.run(eval_genomes, Number_Generations)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Training completed! Best genome saved.")

def main():
    numFiles = count_files(checkpoint_path)

    if os.path.exists("best_genome.pkl"):
        check1 = True
        while(check1):
            answer = input("A trained model was found. Do you want to test it? (yes/no): ").strip().lower()
            if answer == "yes":
                check1 = False
                
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
                return
        
            elif answer == "no":
                check1 = False
                check2 = True
                while(check2):
                    answer2 = input("Do you want to delete the best_genome?").strip().lower()
                    if answer2 == "yes":
                        os.remove("best_genome.pkl")
                        check2 = False
                    elif answer2 == "no":
                        check2 = False
                    else:
                        print("Wrong answer! It is either 'yes' or 'no'")           
            else:
                print("Wrong answer! It is either 'yes' or 'no'")
                
    if numFiles == 0:
        print("No checkpoints found. Starting new training session.")
        run_neat(config_path)
        return

    print("Checkpoint files detected.")
    check3 = True
    while(check3):
        answer = input("Continue training from last checkpoint? If you choose 'no', all checkpoint files will be deleted:").strip().lower()
        if answer == "yes":
            check3 = False
            run_neat(config_path)
        elif answer == "no":
            check3 = False
            
            if os.path.exists(checkpoint_path):
                for file in os.listdir(checkpoint_path):
                    file_path = os.path.join(checkpoint_path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                print("Deleted all checkpoint files ")
                        
            if os.path.exists(robot_info_path):
                try:
                    open(robot_info_path, "w").close()
                except Exception as e:
                    print(f"Error emptying 'robot_info' file: {e}")
                print("Emptied 'robot_info' file")
            
            if os.path.exists(graphs_path):
                for file in os.listdir(graphs_path):
                    file_path = os.path.join(graphs_path, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                print("Deleted all the Graphs")
            
            print("Starting a fresh training.")            
            run_neat(config_path)
        
        else:
            print("Wrong answer! You can only answer with 'yes' or 'no'")

if __name__ == '__main__':
    main()
