from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from scan_simulator_2d import PyScanSimulator2D

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray

from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

import numpy as np
import math
import time

assert rclpy

import time

import tf2_ros
import tf_transformations
import geometry_msgs
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.get_logger().info('symlink check')

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

        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.num_beams_per_particle = self.sensor_model.num_beams_per_particle
        self.scan_field_of_view = self.sensor_model.scan_field_of_view
        self.scan_theta_discretization = self.sensor_model.scan_theta_discretization

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

        self.previous_time = time.perf_counter()

        self.br = tf2_ros.TransformBroadcaster(self)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.dist_error_pub = self.create_publisher(Float32, '/distance_error', 10)
        self.angle_error_pub = self.create_publisher(Float32, '/angle_error', 10)

        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().integer_value


    def publish_avg_pose(self):
        self.publish_particle_poses()
        particle_samples = self.particle_positions

        positions = particle_samples[:, :2]
        angles = particle_samples[:, 2]

        average_position = np.mean(positions, axis=0)
        average_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        
        average_pose = np.hstack((average_position, average_angle))
        self.average_pose = average_pose

        msg = Odometry()

        msg.header.frame_id = '/map'
        
        msg.pose.pose.position.x = average_pose[0]
        msg.pose.pose.position.y = average_pose[1]

        # rotation is around z-axis
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(average_angle / 2)
        msg.pose.pose.orientation.w = np.cos(average_angle / 2)

        msg.child_frame_id = '/base_link'
        # self.get_logger().info("%s" % average_pose)
        # self.get_logger().info("%s" % average_angle)
        self.odom_pub.publish(msg)

        
        # Publish Transform

        obj = geometry_msgs.msg.TransformStamped()

        # current time
        # obj.header.stamp = time.to_msg()

        # frame names
        obj.header.frame_id = '/map'
        obj.child_frame_id = '/base_link_pf'

        # translation component
        obj.transform.translation.x = average_pose[0]
        obj.transform.translation.y = average_pose[1]
        obj.transform.translation.z = 0.0

        # rotation (quaternion)
        obj.transform.rotation.x = 0.0
        obj.transform.rotation.y = 0.0
        obj.transform.rotation.z = np.sin(average_angle / 2)
        obj.transform.rotation.w = np.cos(average_angle / 2)

        self.br.sendTransform(obj)

        try:
            tf_world_to_car: TransformStamped = self.tfBuffer.lookup_transform('map', 'base_link',
                                                                                 rclpy.time.Time())
            
            x_expected = tf_world_to_car.transform.translation.x
            y_expected = tf_world_to_car.transform.translation.y
            angle_expected = 2 * np.arctan2(tf_world_to_car.transform.rotation.z, tf_world_to_car.transform.rotation.w)

            distance_error_msg = Float32()
            distance_error_msg.data = np.sqrt((x_expected - average_pose[0])**2 + (y_expected - average_pose[1])**2)
            
            angle_error_msg = Float32()
            angle_error_msg.data = angle_expected - average_angle

            self.dist_error_pub.publish(distance_error_msg)
            self.angle_error_pub.publish(angle_error_msg)

        except tf2_ros.TransformException:
            self.get_logger().info('no transform from world to base_link_gt found')
            return

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

        downsampled_indices = np.linspace(0, len(msg.ranges)-1, self.num_beams_per_particle).astype(int)
        downsampled_laser_ranges = np.array(msg.ranges)[downsampled_indices]

        weights = self.sensor_model.evaluate(self.particle_positions, downsampled_laser_ranges)
        if weights is None:
            # print("no weights") 
            return

        if np.sum(weights) == 0: return

        weights /= np.sum(weights) # normalize so they add to 1

        # number of particles to keep
        keep = int(0.975*self.num_particles)

        # prevent error
        if np.count_nonzero(weights) < keep: return

        # sample without replacement (`keep` number of particles)
        particle_samples_indices = np.random.choice(len(weights), size=keep, p=weights, replace=False)


        # # for new particles, draw `number of particles` - `keep` particles and add some noise to them
        # repeat_particle_samples_indices = np.random.choice(M, size=self.num_particles - keep, p=weights) 
        # new_particles = self.particle_positions[repeat_particle_samples_indices, :] \
        #                                         + np.random.normal(0.0, 0.01, (self.num_particles - keep, 3))
        # self.particle_positions = np.vstack((self.particle_positions[particle_samples_indices, :], \
        #                                     new_particles))       

        if self.average_pose is not None:
            new_particles = np.tile(self.average_pose, (self.num_particles - keep, 1))
            noise = np.random.normal(0.0, 0.01, size=np.shape(new_particles))
            new_particles += noise
            # update particles       
            self.particle_positions = np.vstack((self.particle_positions[particle_samples_indices, :], \
                                                new_particles))

        self.publish_avg_pose()

    def pose_callback(self, msg):

        self.get_logger().info("initial pose")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.average_pose = np.array([x, y, angle])

        self.get_logger().info(f"initial pose set: {x}, {y}, {angle}")

        # Initialize particles with normal distribution around the user-specified pose
        def normalize_angle(angle):
            # Normalize the angle to be within the range [-π, π]
            normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
            return normalized_angle
        
        x = np.random.normal(loc=x, scale=1.0, size=(self.num_particles,1))
        y = np.random.normal(loc=y, scale=1.0, size=(self.num_particles,1))
        theta = np.random.normal(loc=angle, scale=1.0, size=(self.num_particles,1))

        # Normalize angles
        theta = np.apply_along_axis(normalize_angle, axis=0, arr=theta)

        self.particle_positions = np.hstack((x, y, theta))
        # self.get_logger().info("self.particle_positions: %s" % self.particle_positions)
            
        self.publish_particle_poses()

    def publish_particle_poses(self):
        poses = []
        for i in range(len(self.particle_positions)):
            particle = self.particle_positions[i, :]
            msg = Pose()
            
            msg.position.x = particle[0]
            msg.position.y = particle[1]

            # rotation is around z-axis
            angle = particle[2]
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = np.sin(angle / 2)
            msg.orientation.w = np.cos(angle / 2)

            poses.append(msg)

        msg = PoseArray()
        msg.header.frame_id = '/map'

        msg.poses = poses

        self.particles_pub.publish(msg)


        


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()