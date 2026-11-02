cmake_minimum_required(VERSION 3.0.2)
project(cv_tasks)

find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs cv_bridge)

# installs src/cv_tasks as an importable python package
catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/video_publisher.py
  scripts/shape_subscriber.py
  scripts/color_subscriber.py
  scripts/diamond_subscriber.py
  scripts/denoise_subscriber.py
  scripts/hollow_knight_subscriber.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
