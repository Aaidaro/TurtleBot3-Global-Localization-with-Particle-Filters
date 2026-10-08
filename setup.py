from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'lidar_analysis_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mcmc_pf_localization = lidar_analysis_py.mcmc_particle_filter_node:main',
            'basic_pf_localization = lidar_analysis_py.basic_particle_filter_node:main',
            'pf_error_plotter = lidar_analysis_py.pf_error_plotter:main',
            'pf_explore_motion = lidar_analysis_py.pf_explore_motion_node:main',
            'randomize_robot_pose = lidar_analysis_py.randomize_robot_pose:main',
            'calibrate_truth_transform = lidar_analysis_py.calibrate_truth_transform:main',
            'pf_localization = lidar_analysis_py.particle_filter_node:main',
		'lidar_subscriber = lidar_analysis_py.lidar_subscriber:main',
		'lidar_noise_node = lidar_analysis_py.lidar_noise_node:main',
		'velocity_motion_node = lidar_analysis_py.velocity_motion_node:main',
		'square_odom_recorder = lidar_analysis_py.square_odom_recorder:main',
        ],
    },
)
