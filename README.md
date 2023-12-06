# PANDA Robotic Gripper


For a complete project description, please visit [my personal website](https://www.jason-friedman.me/projects/stacker).

Through Intro to Robotics (MEAM-520), I worked with three classmates to implement and test a series of algorithms which  collectively sorts and stacks blocks of various orientations. Each block has 1 side colored in white, which received extra points if facing upwards in the stack. Points were awarded for each successive vertical block, yet deducted for risky maneuvers causing collapse. We tested our code in simulation and in real life on a seven-joint Franka Emika Panda Robot arm, shown on the right. Ultimately, the manipulator performed these tasks in competition, parallel with another team, thus time and efficiency were critical. 

Throughout the months leading up to our competition, I implemented, unit tested, and simulated various Python algorithms critical to task completion. First I configured the ROS+Gazebo simulation environment on a VirtualBox virtual machine. I subsequently implemented forward kinematics (FK) using Python’s NumPy library and various homogenous transformation matrices, to conveniently convert joint angles into end effector poses in real world coordinates.

Next came inverse velocity kinematics (IVK), using the Jacobian matrix to determine what joint angle differential could achieve any given end effector motion. I then implemented generalized inverse kinematics (IK), by first starting with an initial guess (i.e. “seed”) configuration, then incrementally using IVK to inch towards the goal end effector pose; I even leveraged the over-actuated (7 DoF > 6-D) nature of the system to ensure joint centering within their respective ranges of motion, and avoid reaching joint limits. 

Importantly, path planning expanded upon IK by avoiding colliding with obstacles. In addition to accounting for each linkage’s non-zero cross-sectional area, we wrote RRT and A* algorithms to create non-linear trajectories from initial to goal end effector pose. We then intuitively and experimentally proved that RRT would fit our application best, as it prioritizes runtime over trajectory optimization while still almost always finding valid paths.

During the final weeks before the competition, we integrated all the aforementioned algorithms and wrote some new application-specific functionality. Namely, we first detected all the blocks and their orientations using some basic CV for April tag detection. Next, we wrote helper functions to rotate, pick up, and place each block, and importantly, to evaluate which particular motion sequence must occur given block location and orientation. We finally strung these helper functions together, starting from the closest block for further collision avoidance… And voila! A perfect stack in real-life.