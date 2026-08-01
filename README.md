Open a new terminal and follow the following steps for building Cranium and running Gazebo Simulation.

cd ~/cognipilot/cranium/
colcon build

It will start building 16 packages. Once building is complete, start fresh terminal. In case you face any error in colcon build, this means src folder is not right, follow from Setup Environment again.

source ~/cognipilot/cranium/install/setup.bash
ros2 launch b3rb_gz_bringup sil.launch.py world:=Raceway_1


Lane Vector Extractor:

source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower vectors

Sign Board Classifier:

source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower detect


QR Scanner Node:

source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower qr_detect

Runner Node.

source ~/cognipilot/cranium/install/setup.bash
ros2 run b3rb_ros_line_follower runner
