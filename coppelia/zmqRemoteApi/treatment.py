import os
import neat
import statistics
import matplotlib.pyplot as plt

Checkpoint_Dir = "checkpoints"
Output_File = "output/robot_info"
Output_Graphs_Dir = "output"

os.makedirs(Output_Graphs_Dir, exist_ok=True)

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def main():
    num_files = count_files(Checkpoint_Dir)

    if num_files == 0:
        print("No checkpoint files found.")
        return

    avg_fitnesses = []
    std_fitnesses = []
    best_fitnesses = []
    generations = []
    checkpoints_treated = 0

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

                avg_fitnesses.append(avg_fitness/1000)
                std_fitnesses.append(std_fitness/1000)
                best_fitnesses.append(best_fitness/1000)
                generations.append(pop.generation)

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

                checkpoints_treated += 1

            except Exception as e:
                print(f"Error reading checkpoint {checkpoint_path}: {e}")
                
    print(f"{checkpoints_treated} Checkpoint File{'s' if checkpoints_treated != 1 else ''} Treated. Information dumped in '/output/robot_info'")
    
    plot_graph(generations, avg_fitnesses, "Average Fitness Through Generations", "Generation", "Average Fitness", "avg_fitness.png")
    plot_graph(generations, std_fitnesses, "Standard Deviation Through Generations", "Generation", "Standard Deviation", "std_deviation.png")
    plot_graph(generations, best_fitnesses, "Best Fitnesses Through the Generations", "Generation", "Best Fitnesses", "best_fitnesses.png")
    print("Graphs Made and saved in '/output'")

    
def plot_graph(x_data, y_data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.plot(x_data, y_data, marker='o', color='b', label=ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(Output_Graphs_Dir, filename))
    plt.close()

if __name__ == '__main__':
    main()