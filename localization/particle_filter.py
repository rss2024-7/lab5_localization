from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

import numpy as np
import math
import time

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value


        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.particle_positions = np.array([])
        self.laser_ranges = np.array([])

        # the more often the value i appears in particle_samples_indices, the higher
        # the weight of the pose particle_positions[i]
        self.particle_samples_indices = np.array([])

        self.previous_time = time.perf_counter()

    def publish_avg_pose(self):
        particle_samples = self.particle_positions
        if len(self.particle_samples_indices) != 0:
            particle_samples = self.particle_positions[self.particle_samples_indices, :]

        positions = particle_samples[:, :2]
        angles = particle_samples[:, 2]

        average_position = np.mean(positions, axis=0)
        average_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        
        average_pose = np.hstack((average_position, average_angle))

        msg = Odometry()

        msg.header.frame_id = '/map'
        
        msg.pose.pose.position.x = average_pose[0]
        msg.pose.pose.position.y = average_pose[1]

        # rotation is around z-axis
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(average_angle / 2)
        msg.pose.pose.orientation.w = np.cos(average_angle / 2)

        print(f"estimated_pose: {average_pose[0]}, {average_pose[1]}, {average_angle}")

        msg.child_frame_id = '/base_link'
        # self.get_logger().info("%s" % average_pose)
        # self.get_logger().info("%s" % average_angle)
        self.odom_pub.publish(msg)

    def odom_callback(self, msg):
        if len(self.particle_positions) == 0: return

        # manually time dt
        current_time = time.perf_counter()  # maybe rclpy.time.Time()?
        dt = current_time-self.previous_time

        # relative to robot frame
        x_vel = msg.twist.twist.linear.x
        y_vel = msg.twist.twist.linear.y
        angle_vel = msg.twist.twist.angular.z
        
        odometry = np.array([x_vel * dt, y_vel * dt, angle_vel * dt])

        # print(self.particle_positions)
        self.particle_positions = self.motion_model.evaluate(self.particle_positions, odometry)

        self.publish_avg_pose()

        self.previous_time = current_time

    def laser_callback(self, msg):
        if len(self.particle_positions) == 0: return
        laser_ranges = np.random.choice(np.array(msg.ranges), 100)
        weights = self.sensor_model.evaluate(self.particle_positions, laser_ranges)
        if weights is None:
            # print("no weights") 
            return

        M = len(weights)
        weights /= np.sum(weights) # normalize so they add to 1
        self.particle_samples_indices = np.random.choice(M, size=M, p=weights)

        self.particle_positions = self.particle_positions[self.particle_samples_indices]

        self.publish_avg_pose()

    def pose_callback(self, msg):
        """
        Called every time user manually sets the robot's pose
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = msg.pose.pose.orientation.w

        self.get_logger().info(f"initial pose set: {x}, {y}, {angle}")

        # Initialize particles with normal distribution around the user-specified pose
        def normalize_angle(angle):
            # Normalize the angle to be within the range [-π, π]
            normalized_angle = (angle + math.pi) % (2 * math.pi) - math.pi
            return normalized_angle
        
        x = np.random.normal(loc=x, scale=1.0, size=(self.num_particles,1))
        y = np.random.normal(loc=y, scale=1.0, size=(self.num_particles,1))
        theta = np.random.normal(loc=angle, scale=1.0, size=(self.num_particles,1))

        # Normalize angles
        theta = np.apply_along_axis(normalize_angle, axis=0, arr=theta)

        self.particle_positions = np.hstack((x, y, theta))
        self.get_logger().info("self.particle_positions: %s" % self.particle_positions)


        


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()