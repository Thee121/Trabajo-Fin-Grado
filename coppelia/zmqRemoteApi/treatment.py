import os
import neat
import statistics

Checkpoint_Dir = "checkpoints"
Output_File = "output/robot_info"

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def main():
    num_files = count_files(Checkpoint_Dir)

    if num_files == 0:
        print("No checkpoint files found. Exiting treatment.")
        return

    # Clear the output file if it exists
    with open(Output_File, "w") as out_file:
        out_file.write("Checkpoint Summary:\n\n")

    for i in range(num_files):
        checkpoint_path = os.path.join(Checkpoint_Dir, f"neat_checkpoint-{i}")
        if os.path.exists(checkpoint_path):
            try:
                pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
                
                fitnesses = [g.fitness for g in pop.population.values() if g.fitness is not None]
                avg_fitness = statistics.mean(fitnesses) if fitnesses else 0
                best_fitness = max(fitnesses) if fitnesses else 0
                std_fitness = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0
                extinct = len(pop.species.species) == 0

                with open(Output_File, "a") as out_file:
                    out_file.write(f"Checkpoint: neat_checkpoint-{i}\n")
                    out_file.write(f"Generation: {pop.generation}\n")
                    out_file.write(f"Total species: {len(pop.species.species)}\n")
                    out_file.write(f"Total genomes: {len(pop.population)}\n")
                    out_file.write(f"Average Fitness: {avg_fitness:.4f}\n")
                    out_file.write(f"Best Fitness: {best_fitness:.4f}\n")
                    out_file.write(f"Standard Deviation: {std_fitness:.4f}\n")
                    out_file.write(f"Total Extinction: {'Yes' if extinct else 'No'}\n")
                    out_file.write("-" * 40 + "\n\n")

            except Exception as e:
                print(f"Error reading checkpoint {checkpoint_path}: {e}")
    
if __name__ == '__main__':
    main()
