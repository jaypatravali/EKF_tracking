import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt

np.random.seed(123)
plt.show()
plt.ion()


def generate_measurements(vis=False):
    """
    Generate noisy measurements
    from Ground Truth Data
    """

    vals = scipy.io.loadmat('/home/jay/tracking/data/object_trajectory.mat')['x_traj_pos']
    sigma_r = 0.1
    sigma_a = 3*(np.pi/180)
    for k in range(100):
        f = open("/home/jay/tracking/data/sample_{}.dat".format(k), 'w')
        for i in range(500):
            range_val = np.sqrt(vals[0][i]**2 + vals[1][i]**2)
            noisy_r = range_val + np.random.normal(0, sigma_r)
            azimuth = math.atan2(vals[1][i],vals[0][i])
            noisy_a = azimuth + np.random.normal(0, sigma_a)
            sensor_reading = "{} {}\n".format(noisy_r, noisy_a)
            f.write(sensor_reading)
            if vis:
                p1, = plt.plot(noisy_r*math.cos(noisy_a), noisy_r*math.sin(noisy_a), marker='o', markersize=3, color="red")
                p2, = plt.plot(vals[0][i], vals[1][i],  marker='o', markersize=3, color="blue")
                plt.legend([p1, p2], ['Noisy Path from measurements', 'Ground truth'])
                plt.xlabel("X Axis | Timesteps")
                plt.ylabel("Y axis ")
        plt.show("hold")


def main():
    generate_measurements(False)


if __name__ == "__main__":
    main()
