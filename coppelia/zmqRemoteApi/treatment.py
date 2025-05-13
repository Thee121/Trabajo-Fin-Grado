import os
import neat
import statistics
import matplotlib.pyplot as plt

GRAPH_EVERY_X_GENERATIONS = 1

Checkpoint_Dir = "output/checkpoints"
Output_File = "output/robot_info"
Output_Graphs_Dir = "output/graphs"

os.makedirs(Output_Graphs_Dir, exist_ok=True)

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def main():
    num_files = count_files(Checkpoint_Dir)

    if num_files == 0:
        print("No checkpoint files found.")
        print("Exiting program")
        return

    avg_fitnesses = []
    std_fitnesses = []
    best_fitnesses = []
    generations = []
    lowest_fitnesses = []
    
    checkpoints_treated = 0

    with open(Output_File, "w") as out_file:
        out_file.write("Checkpoint Summary:\n\n")
        out_file.write("-" * 40 + "\n")

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
                    lowest_fitness = min(fitnesses) if fitnesses else 0
                    gen = pop.generation
                    if gen % GRAPH_EVERY_X_GENERATIONS == 0:
                        generations.append(gen)
                        avg_fitnesses.append(avg_fitness)
                        std_fitnesses.append(std_fitness)
                        best_fitnesses.append(best_fitness)
                        lowest_fitnesses.append(lowest_fitness)
                        
                    out_file.write(f"Checkpoint: neat_checkpoint-{i}\n")
                    out_file.write(f"Generation: {gen}\n")
                    out_file.write(f"Total species: {len(pop.species.species)}\n")
                    out_file.write(f"Total genomes: {len(pop.population)}\n")
                    out_file.write(f"Average Fitness: {avg_fitness:.4f}\n")
                    out_file.write(f"Best Fitness: {best_fitness:.4f}\n")
                    out_file.write(f"Lowest Fitness: {lowest_fitness:.4f}\n")
                    out_file.write(f"Standard Deviation: {std_fitness:.4f}\n")
                    out_file.write(f"Total Extinction: {'Yes' if extinct else 'No'}\n")
                    out_file.write("-" * 40 + "\n\n")
                    
                    if len(pop.species.species) > 1:
                                out_file.write(f"Species Breakdown:\n")
                                for sid, _ in pop.species.species.items():
                                    s_fitnesses = [m.fitness for mid, m in pop.population.items() if pop.species.get_species_id(mid) == sid and m.fitness is not None]
                                    s_avg = statistics.mean(s_fitnesses) if s_fitnesses else 0
                                    s_best = max(s_fitnesses) if s_fitnesses else 0
                                    s_lowest = min(s_fitnesses) if s_fitnesses else 0
                                    s_std = statistics.stdev(s_fitnesses) if len(s_fitnesses) > 1 else 0
                                    out_file.write(f"  - Species {sid}: Avg = {s_avg:.4f}, Best = {s_best:.4f}, Lowest = {s_lowest:.4f}, Std = {s_std:.4f}\n")
                                    out_file.write("-" * 40 + "\n\n")
                    checkpoints_treated += 1
                    
                except Exception as e:
                    print(f"Error reading checkpoint {checkpoint_path}: {e}")


    print(f"{checkpoints_treated} Checkpoint File{'s' if checkpoints_treated != 1 else ''} Treated. Information dumped in '/output/robot_info'")

    plot_graph(generations, avg_fitnesses, "Average Fitness Through Generations", "Generation", "Average Fitness", "avg_fitness.png")
    plot_graph(generations, std_fitnesses, "Standard Deviation Through Generations", "Generation", "Standard Deviation", "std_deviation.png")
    plot_graph(generations, best_fitnesses, "Best Fitness Through Generations", "Generation", "Best Fitness", "best_fitnesses.png")
    plot_graph(generations, lowest_fitnesses, "Lowest Fitness Through Generations", "Generation", "Lowest Fitness", "lowest_fitnesses.png")

    plot_combined_graph(generations, avg_fitnesses, std_fitnesses, best_fitnesses, lowest_fitnesses)

    print("Graphs Made and saved in '/output/Graphs'")
    print("Exiting program")

def plot_graph(x_data, y_data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(20, 10))
    plt.plot(x_data, y_data, marker='o', color='b', label=ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(Output_Graphs_Dir, filename))
    plt.close()

def plot_combined_graph(gens, avg, std, best, lowest, filename="combined_metrics.png"):
    plt.figure(figsize=(20, 10))
    plt.plot(gens, avg, label='Avg Fitness', color='blue', marker='o')
    #plt.plot(gens, std, label='Std Deviation', color='orange', marker='s')
    plt.plot(gens, best, label='Best Fitness', color='green', marker='d')
    plt.plot(gens, lowest, label='Lowest Fitness', color='yellow', marker='d')


    plt.title("Combined Metrics Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(Output_Graphs_Dir, filename))
    plt.close()

if __name__ == '__main__':
    main()
