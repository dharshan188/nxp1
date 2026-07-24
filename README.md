# Running the B3RB ROS2 Line Follower

## Terminal 1 – Launch Simulation

```bash
ros2 launch b3rb_gz_bringup sil.launch.py world:=Raceway_1
```

---

## Terminal 2 – Run Edge Vector Detection

```bash
cd ~/cognipilot/cranium

source install/setup.bash

ros2 run b3rb_ros_line_follower b3rb_ros_edge_vectors
```

---

## Terminal 3 – Run Line Follower Controller

```bash
cd ~/cognipilot/cranium

source install/setup.bash

ros2 run b3rb_ros_line_follower b3rb_ros_line_follower
```
