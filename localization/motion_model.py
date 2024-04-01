import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        def get_noise(std_x, std_y, std_theta, shape):
            std = np.array(std_x, std_y, std_theta, shape)
            noise = np.random.normal(0, std, shape)
            return noise
        
        def get_transform(arr):
            # returns given 1x3 [x,y,theta] vector, returns correspondinng 3x3 transformation matrix
            x = arr[0]
            y = arr[1]
            theta = arr[2]
            transform = np.array([[np.cos(theta), -np.sin(theta), x],
                      [np.sin(theta), np.cos(theta), y],
                      [0, 0, 1]])
            return transform
        
        def get_pose(transform):
            # given 3x3 transformation matrix, returns 1x3 [x,y,theta] vector
            x = transform[0][2]
            y = transform[1][2]
            theta = np.arccos(transform[0][0])

            return np.array([x,y,theta])

        particles_std_x = 1
        particles_std_y = 1
        particles_std_theta = 1
        particles_noise = get_noise(particles_std_x, particles_std_y,particles_std_theta,particles.shape)
        particles += particles_noise

        odometry_std_x = 1
        odometry_std_y = 1
        odometry_std_theta = 1
        odometry_noise = get_noise(odometry_std_x, odometry_std_y, odometry_std_theta,odometry.shape)
        odometry += odometry_noise

        transform_particles = np.apply_along_axis(get_transform, 1, particles)
        transform_odometry = get_transform(odometry)

        transform_new_particles = transform_particles @ transform_odometry

        # does same thing as new_particles = np.array([[get_pose(row) for row in transform_new_particles]]) 
        new_particles = np.zeros(particles.shape)
        new_particles[:][:,0] = transform_new_particles[:][:,0][:,2]
        new_particles[:][:,1] = transform_new_particles[:][:,1][:,2]
        new_particles[:][:,2] = np.arccos(transform_new_particles[:][:,0][:,0])

        
        return new_particles
        ####################################
