import numpy as np
from scan_simulator_2d import PyScanSimulator2D
from mpl_toolkits.mplot3d import Axes3D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

import matplotlib.pyplot as plt

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def plot_sensor_model_table(self):
        # Assuming self.sensor_model_table is your 201x201 numpy array
        sensor_model_table = self.sensor_model_table

        # Create x and y indices
        x = np.arange(sensor_model_table.shape[0])
        y = np.arange(sensor_model_table.shape[1])
        X, Y = np.meshgrid(x, y)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, sensor_model_table, cmap='viridis')

        # Add a color bar which maps values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Set labels
        ax.set_xlabel('True Distance')
        ax.set_ylabel('Measured Distance')
        ax.set_zlabel('Probability')

        plt.show()

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        z_max = self.table_width - 1

        p_hit = np.zeros((self.table_width, self.table_width))
        p_short = np.zeros((self.table_width, self.table_width))
        p_max = np.zeros((self.table_width, self.table_width))
        p_rand = np.full((self.table_width, self.table_width), 1 / z_max)

        z, d = np.indices((self.table_width, self.table_width))

        # Case 1 - probability of detecting known obstacle in map
        eta = 1
        p_hit = np.exp( - (z - d)**2 / (2 * self.sigma_hit**2) )
        p_hit *= eta / np.sqrt(2 * np.pi * self.sigma_hit**2)
        # normalize p_hit
        p_hit /= np.sum(p_hit, axis = 0)
        
        # Case 2 - probability of short measurement
        short_indices = np.where(np.logical_and(z <= d, d != 0))
        p_short[short_indices] = 2/d[short_indices] * (1 - z[short_indices]/d[short_indices])
            
        # Case 3 - probability of large measurement
        p_max[np.where(z == z_max)] = 1
            
        # Case 4 - probability of completely random measurement
        # p_rand = 1 / z_max

        self.sensor_model_table = self.alpha_hit * p_hit + self.alpha_short * p_short + self.alpha_max * p_max + self.alpha_rand * p_rand

        # normalize entire sensor model (So probabilities sum to 1)
        self.sensor_model_table /= np.sum(self.sensor_model_table, axis=0)

        # self.plot_sensor_model_table()
    
        return


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^(i)

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        # n (number of particles) x num_beams_per_particle
        scans = self.scan_sim.scan(particles)
        
        # convert units from meters to pixels
        scans /= self.resolution  * self.lidar_scale_to_map_scale
        observation /= self.resolution  * self.lidar_scale_to_map_scale

        # clip values (cap their values at z_max) in scans matrix
        z_max = self.table_width - 1
        observation = np.floor(np.clip(observation, 0, z_max)).astype(int)
        scans = np.floor(np.clip(scans, 0, z_max)).astype(int)

        # Index the sensor model table with observation to get a 100 x 201 array
        # Then, use scans to select the relevant probabilities for each particle
        # This results in an n x 100 array where each row corresponds to a particle
        # selected_probabilities = self.sensor_model_table[observation, :][:, scans][0]


        probabilities = np.ones(len(particles))
        for i in range(scans.shape[0]):  # for each particle
            for j in range(scans.shape[1]):  # for each beam in the scan
                # Look up the probability from the sensor model table
                prob = self.sensor_model_table[scans[i, j], observation[j]]
                
                # Multiply the probabilities to get a cumulative likelihood for the particle
                probabilities[i] *= prob

        return probabilities






        probabilities = []
        for i in range(np.shape(scans)[0]):
            probabilities_for_particle_i = self.sensor_model_table[scans[i], observation]
            probabilities.append(probabilities_for_particle_i)
        probabilities = np.array(probabilities)

        print(f"probabilities: {probabilities}")

        # Get total log probability for each particle (equivalent to taking product of log of probabilities of each scan)
        # length n vector containing probability of each particle being correct
        likelihoods = np.prod(probabilities, axis=1)
        # likelihoods = np.sum(np.log(selected_probabilities), axis=1)
        # likelihoods = np.sum(selected_probabilities, axis=1) / self.num_beams_per_particle
        return likelihoods 


        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution  # 0.05040000006556511

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
