# Robotica

This repository provides software tools for robots.  It includes [Python3](https://www.python.org/) code for managing the communication with robotic simulators such as [CoppeliaSim](https://www.coppeliarobotics.com/) and some samples of wheeled robot controllers.

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

The main class contains basic functionality aside from running the main function of the project. The program detects if there are checkpoints or a best_genome file present.

it will start a fresh NEAT training session.

All the output of the neat library are dumped in the neat_output file after each checkpoint. Only until all the generations are ran and a best_genome is generated, then the code will automatically clean and present a better version in the neat_output_clean file, deleting the original neat_output.

This functionality is present such that the user does not have to run all the generations in one go or if there is a problem, it is able to resume from the latest checkpoint possible. It is noted that the neat_config OR the modifiable variables can not be changed in between continuations of generations. In other words the program will fail if it detects a change in the settings or modifiable variables and will only continue training if the values are changed to match.