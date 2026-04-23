"""
Minimal UR10e launch with fake hardware — no urscript_interface.
Avoids the "Failed to connect to robot" error.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    # Include the standard UR control launch but without RViz
    # (we launch RViz separately to avoid urscript issues)
    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'launch', 'ur_control.launch.py'
            ])
        ),
        launch_arguments={
            'ur_type': 'ur10e',
            'robot_ip': '0.0.0.0',
            'use_fake_hardware': 'true',
            'launch_rviz': 'false',
            'initial_joint_controller': 'forward_velocity_controller',
            'launch_dashboard_client': 'false',
            'headless_mode': 'true',
        }.items(),
    )

    # Kill the urscript_interface after launch (it has no condition flag)
    kill_urscript = ExecuteProcess(
        cmd=['bash', '-c', 'sleep 5 && pkill -f urscript_interface && pkill -f trajectory_until; echo "Killed urscript_interface"'],
        output='screen',
    )

    # Launch RViz with UR description
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('ur_description'),
            'rviz', 'view_robot.rviz'
        ])],
    )

    return LaunchDescription([
        ur_launch,
        kill_urscript,
        rviz_node,
    ])
