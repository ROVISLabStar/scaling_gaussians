"""
Launch file for UR10e Visual Servoing with 3DGS.

Launches:
1. UR10e driver with fake hardware
2. Forward velocity controller
3. VS node with gsplat rendering
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    ExecuteProcess, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Arguments ----
    ckpt_arg = DeclareLaunchArgument('ckpt', description='Path to gsplat checkpoint')
    cfg_arg = DeclareLaunchArgument('cfg', description='Path to gsplat config')
    goal_arg = DeclareLaunchArgument('goal_idx', default_value='10')
    mode_arg = DeclareLaunchArgument('mode', default_value='inflated')
    scale_arg = DeclareLaunchArgument('scale_factor', default_value='1.8')
    gain_arg = DeclareLaunchArgument('gain', default_value='10.0')

    # ---- UR10e with fake hardware ----
    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'),
                'launch', 'ur_control.launch.py'
            ])
        ),
        launch_arguments={
            'ur_type': 'ur10e',
            'use_fake_hardware': 'true',
            'launch_rviz': 'true',
            'initial_joint_controller': 'forward_velocity_controller',
        }.items(),
    )

    # ---- VS Node (delayed to let UR driver start) ----
    vs_node = TimerAction(
        period=5.0,  # Wait 5 seconds for UR driver to start
        actions=[
            Node(
                package='gs_vs_ros',
                executable='vs_node.py',
                name='vs_node',
                output='screen',
                parameters=[{
                    'ckpt': LaunchConfiguration('ckpt'),
                    'cfg': LaunchConfiguration('cfg'),
                    'goal_idx': LaunchConfiguration('goal_idx'),
                    'mode': LaunchConfiguration('mode'),
                    'scale_factor': LaunchConfiguration('scale_factor'),
                    'gain': LaunchConfiguration('gain'),
                }],
            ),
        ],
    )

    return LaunchDescription([
        ckpt_arg, cfg_arg, goal_arg, mode_arg, scale_arg, gain_arg,
        ur_launch,
        vs_node,
    ])
