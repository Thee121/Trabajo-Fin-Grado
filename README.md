# Robotica

This repository provides software tools for robots.  It includes [Python3](https://www.python.org/) code for managing the communication with robotic simulators such as [CoppeliaSim](https://www.coppeliarobotics.com/).

The repository can be downloaded and used as needed.  It is encouraged to maintain the original directory structure as in the repository.  It will be described any specific dependency between modules or scripts.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

This repository is a fork of https://github.com/jdlope/robotica.git

## Description

This is and end of degree project for Computer Engineering in the Universidad Politécnica de Madrid.

The current implementation of the brain has two different modes:

1) FollowLine: main function of the brain, prioritizing it over the other mode of thinking. It slowly learns how to follow a line, learning how to better turn sharp turns and keeping the line centered. It uses the camera as input feed, detecting only black lines at the moment.

2) FindingLine: secondary mode of the brain if a line is not found. It learns automatically the best way to search a space and avoid obstacles.

Take into consideration that, how the calculate_fitness() method is implemented, the robot prioritizes following a line over finding the line and avoiding obstacles. This does not mean the robot will not learn how to find the line or avoid obstacles, but it will instead prefer to follow the line over moving randomely around the map.

ALL possible variables that could be modified are easily modifiable, either at the beginning of avoid.py or at neat_config.txt.

The main class contains extra functionality aside from actually running the main function of the project. The program logic is as follows:
1) Detects if a best_genome is present. If there is, asks user if it wants to train it. If the user answers with "yes", the program runs the genome. If the user answers negatively, the program advances to step 2.
2) The program checks if there are any checkpoints in the folder. If there aren't, a new fresh session starts. If there are checkpoints present, the program advances to step 3
3) Asks if you want to continue from the previous checkpoint. If the user answers with "yes", the program will continue from the previous checkpoint. If the users answers "no", a fresh NEAT training session will start

All the checkpoints of the neat library are dumped in the checkpoint directory after each checkpoint. If you want to see the information from all the checkpoints, you can run the treatment.py file to output all the information to robot_info.txt. Several graphs are also generated to better visualize the information over the generations. Except for standard deviation, all other appropiate information is combined to create a single graph to better study the robot's performance over the generations.
