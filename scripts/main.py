import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import argparse

# Pass q scaling factor as argument list.
parser = argparse.ArgumentParser(description='Change Q Value ')
parser.add_argument('-q_list',  type=float, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()

plt.ion()
plt.show()


class EFK(object):
    """
    Extended Kalman filter for Object Tracking.
    """

    def __init__(self, args):
        self.read_groundTruth_data()
        self.init_filter_params(args)
        self.compute_ground_truth_velocity()
        self.vessel_states_dict = {}

    def init_filter_params(self, args):
        """
        initialize filter parameter
        """

        self.sigma_a = (3*np.pi)/180
        self.sigma_r = 0.1
        self.R = np.array([[self.sigma_r**2, 0.0],
                           [0.0,  self.sigma_a**2]])
        self.RMSE_errors = []
        self.q_list = args.q_list

    def read_sensor_data(self, filename):
        """
        Read sensor measurements
        """

        sensor_readings = {}
        f = open(filename)
        timestep = 0
        for line in f:

            line_s = line.split('\n')  # remove the new line character
            line_spl = line_s[0].split(' ')  # split the line
            sensor_readings[timestep] = [float(line_spl[0]), float(line_spl[1])]
            timestep += 1
        return sensor_readings

    def init_sample_params(self, q_val, sample):
        """
        initialize variables for each EKF sample iteration
        """

        self.mu = np.array([[50], [50], [0.0], [0.0]])
        self.P = np.array([[1,    0.0,  0.0, 0.0],
                           [0.0,   1,   0.0, 0.0],
                           [0.0,  0.0,   1,  0.0],
                           [0.0,  0.0,  0.0,  1]])
        self.vessel_states = []
        self.q_factor = q_val
        self.observed_x = []
        self.observed_y = []
        self.EFK_x = []
        self.EFK_y = []
        self.predicted_x = []
        self.predicted_y = []

        sensor_readings = self.read_sensor_data("../data/sample_{}.dat".format(sample))
        return sensor_readings

    def compute_ground_truth_velocity(self):
        """Computes ground truth velocities in x and y
           from ground truth trajectory and 0.1s time-step
        """

        # initial velocity for timestep 0 is assumed as zero
        # from the  given problem statement
        self.gt_vx = [0]
        self.gt_vy = [0]
        for timestep in range(1, len(self.gt_data[0])):
            self.gt_vx.append((self.gt_data[0][timestep] - self.gt_data[0][timestep-1])/0.1)
            self.gt_vy.append((self.gt_data[1][timestep] - self.gt_data[1][timestep-1])/0.1)

    def read_groundTruth_data(self):
        """
        Load Ground Truth Data
        """

        self.gt_data = scipy.io.loadmat('../data/object_trajectory.mat')['x_traj_pos']

    def correction_step(self, timestep, sensor_data):
        """
        Performs Measurement update
         and correction
        """

        X = self.mu[0][0]
        Y = self.mu[1][0]
        total = (X**2 + Y**2)

        H = np.array([[X/math.sqrt(total),  Y/math.sqrt(total), 0.0, 0.0],
                     [-Y/total,             X/total,            0.0,  0.0]])

        h_fn = np.array([[total**0.5],
                         [math.atan2(Y, X)]])

        Z = np.array([[sensor_data[0]],
                      [sensor_data[1]]])

        K_intermediate = np.linalg.inv(np.dot(np.dot(H, self.P), np.transpose(H)) + self.R)
        K = np.dot(np.dot(self.P, np.transpose(H)), K_intermediate)
        self.mu = self.mu + np.dot(K, (Z - h_fn))
        self.P = np.dot(np.eye(len(self.P)) - np.dot(K, H), self.P)

    # def correction_step(self, timestep, sensor_data):
    #     """
    #     Performs Measurement update
    #     and correction.
    #     Express h_fn and H in
    #     X and Y
    #     """

    #     X = self.mu[0][0]
    #     Y = self.mu[1][0]
    #     total = (X**2 + Y**2)

    #     h_fn = np.array([[X],
    #                      [Y]])

    #     H = np.array([[1,  0.0,   0.0, 0.0],
    #                  [0.0,  1,    0.0, 0.0]])

    #     Z = np.array([[sensor_data[0]*np.cos(sensor_data[1])], [sensor_data[0]*np.sin(sensor_data[1])]])

    #     K_intermediate = np.linalg.inv(np.dot(np.dot(H, self.P), np.transpose(H)) + self.R)
    #     K = np.dot(np.dot(self.P, np.transpose(H)), K_intermediate)
    #     self.mu = self.mu + np.dot(K, (Z - h_fn))
    #     self.P = np.dot(np.eye(len(self.P)) - np.dot(K, H), self.P)

    def prediction_step(self, timestep):
        """
        Applies motion model to
        get predicted state and covariance
        """

        T = 0.1
        T_3 = T**3/3
        T_2 = T**2/2

        F = np.array([[1,    0.0,   T,  0.0],
                      [0.0,   1,   0.0,  T],
                      [0.0,  0.0,   1,  0.0],
                      [0.0,  0.0,  0.0,  1]])

        Q = np.array([[T_3,  0.0,  T_2, 0.0],
                      [0.0,  T_3,  0.0, T_2],
                      [T_2,  0.0,   T,  0.0],
                      [0.0,  T_2,  0.0,  T]])

        Q = self.q_factor*Q
        self.mu = np.dot(F, self.mu)
        self.P = np.dot(np.dot(F, self.P), np.transpose(F)) + Q

    def visualize_results(self):
        """
        plots RMSE errors in x,y,vx, vy
        for 3 q_factors values
        """

        fig, axarr = plt.subplots(4, sharex=True)
        fig.suptitle('EFK Root Mean Squared Error Plot', size=16)

        colors = ['b', 'r', 'g']

        for index in range(len(self.q_list)):
            l1, = axarr[0].plot(self.RMSE_errors[index][0], color=colors[index], label='q_factor_{} val: {}'.format(index, self.q_list[index]))
            l2, = axarr[1].plot(self.RMSE_errors[index][1], color=colors[index], label='q_factor_{} val: {}'.format(index, self.q_list[index]))
            l3, = axarr[2].plot(self.RMSE_errors[index][2], color=colors[index], label='q_factor_{} val: {}'.format(index, self.q_list[index]))
            l4, = axarr[3].plot(self.RMSE_errors[index][3], color=colors[index], label='q_factor_{} val: {}'.format(index, self.q_list[index]))

        axarr[0].set_title("RMSE X Coordinate (m)")
        axarr[1].set_title("RMSE Y Coordinate (m)")
        axarr[2].set_title("RMSE Velocity in X (m/s)")
        axarr[3].set_title("RMSE Velocity in Y (m/s)")
        axarr[3].set_xlabel("Timestep -->")

        axarr[0].set_ylabel("X RMSE in m ")
        axarr[1].set_ylabel("Y RSME in m")
        axarr[2].set_ylabel("Vx RSME in m/s")
        axarr[3].set_ylabel("Vy RSME in m/s")

        box = axarr[0].get_position()
        axarr[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axarr[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axarr[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axarr[3].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplots_adjust(left=0.07, bottom=0.08, right=0.77, top=0.84)
        plt.legend()
        plt.show("hold")

    def visualize_path(self, prediction, timestep, sensor_data):
        """
        plots paths from
        EKF.
        """
        x = sensor_data[0]*math.cos(sensor_data[1])
        y = sensor_data[0]*math.sin(sensor_data[1])
        self.predicted_x.append(prediction[0][0])
        self.predicted_y.append(prediction[1][0])
        self.observed_x.append(x)
        self.observed_y.append(y)
        self.EFK_x.append(self.mu[0])
        self.EFK_y.append(self.mu[1])

        if timestep == 499:
            plt.plot(self.predicted_x, self.predicted_y,  marker='o', markersize=3, color="red", label='Prediction State')
            plt.plot(self.observed_x, self.observed_y, marker='o', markersize=2, color="green", label='Path from Measurements')
            plt.plot(self.EFK_x, self.EFK_y, marker='o', markersize=2, color="black", label='EKF Path')
            plt.plot(self.gt_data[0], self.gt_data[1],  marker='o', markersize=2, color="blue", label='Ground Truth')
            plt.xlabel("X Coordinates | Timesteps")
            plt.ylabel("Y Coordinates ")
            plt.legend()
            plt.show("hold")

    def compute_statistics(self):
        """
        Compute RMSE error for a q value
        """

        rmse_X = []
        rmse_Y = []
        rmse_Vx = []
        rmse_Vy = []
        state = self.vessel_states_dict
        for timestep in range(len(state[0])):
            error_sum_x = 0
            error_sum_y = 0
            error_sum_vx = 0
            error_sum_vy = 0
            for sample in range(len(state)):
                error_sum_x += (self.gt_data[0][timestep] - state[sample][timestep][0])**2
                error_sum_y += (self.gt_data[1][timestep] - state[sample][timestep][1])**2
                error_sum_vx += (self.gt_vx[timestep] - state[sample][timestep][2])**2
                error_sum_vy += (self.gt_vy[timestep] - state[sample][timestep][3])**2

            rmse_X += [math.sqrt(error_sum_x/len(state))]
            rmse_Y += [math.sqrt(error_sum_y/len(state))]
            rmse_Vx += [math.sqrt(error_sum_vx/len(state))]
            rmse_Vy += [math.sqrt(error_sum_vy/len(state))]

        self.RMSE_errors.append([rmse_X, rmse_Y, rmse_Vx, rmse_Vy])
        self.vessel_states_dict = {}

    def run_EFK(self):
        """
        Run 100 EFK samples
        """

        print("Running an EKF for 100 Monte Carlo Simulations")
        for q_val in self.q_list:
            for sample in range(1):
                sensor_readings = self.init_sample_params(q_val, sample)
                for timestep in range(0, 500):
                    self.prediction_step(timestep+1)
                    prediction = self.mu
                    self.correction_step(timestep, sensor_readings[timestep])
                    self.vessel_states.append(self.mu)
                    # Uncomment the line below to get plot for trajectories.
                    # self.visualize_path(prediction, timestep, sensor_readings[timestep])
                self.vessel_states_dict[sample] = self.vessel_states
            self.compute_statistics()
        self.visualize_results()


def main():
    EFK(args).run_EFK()


if __name__ == "__main__":
    main()
