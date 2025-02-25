import os
import numpy as np
import scipy as sp #for numerical ODE solver
import matplotlib.pyplot as plt #for visualization
from initialization import initial_conditions, get_initial_theta_prime_0, r_prime_0
from integration import plot_until_xstop, make_circle, get_preimage_set, get_observer_view
from initialization import get_v_prime_0
from scipy.interpolate import splprep, splev


# a function that finds the intersection of the extension of a tanget vector with a line
def find_intersection(last_two_points, b):
  point_1 = last_two_points[0]
  point_2 = last_two_points[1]
  x = (b - point_1[1])*(point_1[0] - point_2[0])/(point_1[1] - point_2[1]) + point_1[0]
  return x

# This function trace out the trajectory of a particle shooting horizontally to the positive x-axis with
# an initial velocity v0 and an impact parameter b, it also return the lensing angle
def impact_parameter_and_lensing_angle(geodesic, impact_parameter, m_prime, m0, mf, check):
    '''
    find the lensing angle of a particle for a given impact parameter
    '''
    # set up initial condition
    initial_conditions_0 = [0, impact_parameter, 0.0000000000001, 0, impact_parameter/np.sqrt(2), 0, 1, 0]
    
    # the expected projection for vaidya geodesic
    initial_trajectory = plot_until_xstop(geodesic, initial_conditions_0, 100, m_prime, m0, mf, check)
    
    if check: 
      # visualize the solution
      fig = plt.figure(figsize=(15, 8))

      # Add a 2D subplot in the second position of a 1x2 grid
      ax = fig.add_subplot(111)
      ax.plot(initial_trajectory[0], initial_trajectory[2], color = "goldenrod")
      # set lables
      ax.set_xlabel("x")
      ax.set_ylabel("z")
      ax.set_xlim(-3,3)
      ax.set_ylim(-5,3)

      # Set the aspect ratio to be equal except the 3rd one
      ax.set_aspect('equal', adjustable='box')
      ax.grid(True)

      # Set titles
      ax.set_title('Trajectory of lensing particle at impact parameter = ' + str(impact_parameter))
      
      # Save fig
      fig.savefig(os.path.join("particle_lensing_images", f"Trajectory of lensing particle at impact parameter = {impact_parameter}.png"), dpi=300, bbox_inches='tight')
      
    # calculate angle
    if np.min(initial_trajectory[0]) < 0:
      return 0
    else:
      last_two_points = np.array([(initial_trajectory[0][-1], initial_trajectory[2][-1]), 
                                 (initial_trajectory[0][-2], initial_trajectory[2][-2])])
      x = find_intersection(last_two_points, impact_parameter)
      angle = np.arctan((initial_trajectory[2][-1] - impact_parameter)/(initial_trajectory[0][-1] - x))
      if angle > 0:
        return 0
      else:
        return - angle * 180 / np.pi


# plot the image of the telescope
def plot_image(radius, points, x, theta, v0, m_prime, m0, mf, geodesic, directory, case_color):
    '''
    Plot telescope image with the given parameters and put it in a specific directory.
    '''
    # due to spherical symmetry we can always set the phi looking direction to 0
    initial_looking_directions = np.array([get_initial_theta_prime_0(r_prime_0, x, theta), 0])
    initial_conditions_0 = initial_conditions(*initial_looking_directions, v0, x, m_prime, m0, mf)

    # the expected projection for vaidya geodesic
    initial_trajectory = plot_until_xstop(geodesic, initial_conditions_0, -x, m_prime, m0, mf)
    circle = make_circle(radius, initial_trajectory[2][-1], initial_trajectory[0][-1], points)

    # initialize set of initial conditions
    set_of_initial_conditions = [initial_conditions_0]

    # define a list of solutions in cartesian coordinate which will be used later
    set_of_trajectories = []

    # extract each point in the circle
    points_in_circle = []
    for i in range(len(circle[:][0])):
      points_in_circle.append((circle[0][i], circle[1][i], circle[2][i]))

    # get set of initial conditions for the circle
    set_of_initial_conditions.extend(get_preimage_set(geodesic, points_in_circle,
                                    [initial_trajectory[0][-1], initial_trajectory[1][-1], initial_trajectory[2][-1]],
                                    initial_conditions_0, v0, x, m_prime, m0, mf))

    #5. Numerically solve the geodesic
    #   args=(mass,) parameter is used to define any constants

    for initial_condition in set_of_initial_conditions:
      set_of_trajectories.append(plot_until_xstop(geodesic, initial_condition, -x, m_prime, m0, mf))


    #6. Visualize the solution
    fig2 = plt.figure(figsize=(15, 8))

    # Add a 2D subplot in the second position of a 1x2 grid
    ax3 = fig2.add_subplot(111)

    # set lables
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")

    # Set the aspect ratio to be equal except the 3rd one
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True)

    # Set titles
    ax3.set_title('Telescope image')

    # plot looking directions on 2D plane
    x_tele_image, y_tele_image = get_observer_view(set_of_trajectories, theta)
    # Parametric interpolation using splprep
    tck, u = splprep([x_tele_image, y_tele_image], s=0, per=True)  # s=0 for smooth interpolation, # per=True for closed curve
    u_new = np.linspace(0, 1, 1000)  # Finer parameter space
    x_new, y_new = splev(u_new, tck)  # Interpolated x and y coordinates
    ax3.plot(x_new, y_new, color = case_color, linewidth=5)  # Interpolated points

    # Save fig2
    fig2.savefig(os.path.join(directory, f"telescope_image.png = {theta}.png"), dpi=300, bbox_inches='tight')


def get_telescope_image(geodesic, theta, radius, r0, points, v0, m_prime, m0, mf):
  # due to spherical symmetry we can always set the phi looking direction to 0
  initial_looking_directions = np.array([get_initial_theta_prime_0(r_prime_0, r0, theta), 0])
  initial_conditions_0 = initial_conditions(*initial_looking_directions, v0, r0, m_prime, m0, mf)

  # the expected projection
  initial_trajectory = plot_until_xstop(geodesic, initial_conditions_0, -r0, m_prime, m0, mf)

  # calculate shooting problem solutions
  set_of_initial_conditions = [initial_conditions_0]
  set_of_trajectories = [initial_trajectory]
  circle = make_circle(radius, initial_trajectory[2][-1], initial_trajectory[0][-1], points)
  points_in_circle = []
  for i in range(len(circle[:][0])):
    points_in_circle.append((circle[0][i], circle[1][i], circle[2][i]))
  set_of_initial_conditions.extend(get_preimage_set(geodesic, points_in_circle,
                                  [initial_trajectory[0][-1], initial_trajectory[1][-1], initial_trajectory[2][-1]],
                                  initial_conditions_0, v0, r0, m_prime, m0, mf))
  for initial_condition in set_of_initial_conditions:
    set_of_trajectories.append(plot_until_xstop(geodesic, initial_conditions_0, -r0, m_prime, m0, mf))

  x_tele_image, y_tele_image = get_observer_view(set_of_trajectories, theta)
  return (x_tele_image, y_tele_image)

# find critical theta
def find_critical_theta(geodesic, radius, r0, v0, m_prime, m0, mf):
  #theta_list = np.linspace(np.pi/10, np.pi/9, 100)
  theta_list = np.linspace(0.32, 0.5, 20)
  #theta_list = np.linspace(0.06, 0.1, 100)
  i = 0
  x_tele_image, y_tele_image = get_telescope_image(geodesic, theta_list[i], radius, r0, 8, v0, m_prime, m0, mf)
  y_min = min(y_tele_image)
  while y_min != y_tele_image[5]:
    i += 1
    x_tele_image, y_tele_image = get_telescope_image(geodesic, theta_list[i], radius, r0, 8, v0, m_prime, m0, mf)
    y_min = min(y_tele_image)
  return theta_list[i]


def get_eccentricity(geodesic, theta, radius, r0, v0, m_prime, m0, mf):
    eccentricity_set = []
    x_tele_image, y_tele_image = get_telescope_image(geodesic, theta, radius, r0, 4, v0, m_prime, m0, mf)
    x_max = max(x_tele_image)
    x_min = min(x_tele_image)
    y_max = max(y_tele_image)
    y_min = min(y_tele_image)
    a = (x_max - x_min) / 2
    b = (y_max - y_min) / 2
    if (x_max - x_min) < (y_max - y_min):
        a = (y_max - y_min) / 2
        b = (x_max - x_min) / 2

    eccentricity = np.sqrt(1 - b**2 / a**2)
    eccentricity_set.append(eccentricity)
    return eccentricity