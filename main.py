# main.py
from integration import plot_until_xstop, vaidya_geodesic, make_circle, get_preimage_set, get_observer_view
from initialization import mass_func, get_initial_theta_prime_0, initial_conditions, theta0, phi0, r_prime_0
import sys
import numpy as np
import matplotlib.pyplot as plt

# lists fixed initial conditions
phi0 = 0
theta0 = np.pi/2
r_prime_0 = -1/np.sqrt(2)

def main(radius, points, x, theta, v0, m_prime, m0, mf):
  geodesic = vaidya_geodesic
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
  fig1 = plt.figure(figsize = (20, 20))
  # Add a 3D subplot in the first position of a 1x2 grid
  ax1 = fig1.add_subplot(111, projection='3d')

  # plot trajectories in 3D.
  for xpoints,ypoints,zpoints in set_of_trajectories:
    ax1.plot3D(xpoints, ypoints, zpoints, "goldenrod")
    ax1.plot3D(xpoints[0],ypoints[0],zpoints[0], 'ro') # plots red circle at initial point
    ax1.plot3D(xpoints[-1],ypoints[-1],zpoints[-1], 'bo') # plots blue circle at end point
    ax1.plot3D([0], [0], [0], 'o', color='black', markersize=20 * mass_func(v0, m_prime, m0, mf)[0]) # the black hole location
  #ax1.set_ylim3d(-10,10)
  ax1.set_xlim3d(-x,x)
  #ax1.set_zlim3d(0,170)

  # set titles
  ax1.set_title('Set of Trajectories')

  # set lables
  ax1.set_xlabel("x")
  ax1.set_ylabel("y")
  ax1.set_zlabel("z")

  fig2 = plt.figure(figsize=(15, 8))
  
  # Add a 2D subplot in the second position of a 1x2 grid
  ax2 = fig2.add_subplot(121)

  # Add a 2D subplot in the second position of a 1x2 grid
  ax3 = fig2.add_subplot(122)

  # plot projections on 2D plane
  for xpoints,ypoints,zpoints in set_of_trajectories:
    ax2.plot(ypoints[-1], zpoints[-1], 'bo') # plots blue circle at end point

  # set lables
  ax3.set_xlabel("y")
  ax3.set_ylabel("z")
  ax2.set_xlabel("y")
  ax2.set_ylabel("z")

  # Set the aspect ratio to be equal except the 3rd one
  ax2.set_aspect('equal')
  ax3.set_aspect('equal', adjustable='box')
  ax2.grid(True)
  ax3.grid(True)

  # Set titles
  ax2.set_title('Projection and Expectation')
  ax3.set_title('Telescope image')

  # plot looking directions on 2D plane
  x_tele_image, y_tele_image = get_observer_view(set_of_trajectories)
  #x_tele_image.pop(0)
  #y_tele_image.pop(0)
  ax3.scatter(x_tele_image, y_tele_image)

  # plot the expected circle in 2D
  ax2.scatter(circle[1], circle[2], color = "purple")

  # Show the plot
  plt.show()

# system modification for the main function to be called from the command line
if __name__ == '__main__':
    # Check if the correct number of arguments was provided
    if len(sys.argv) != 9:
        print("Usage: python3 main.py <radius> <points> <x> <theta> <v0>")
        sys.exit(1)
    
    # Parse command-line arguments
    try:
        radius = float(sys.argv[1])
        points = int(sys.argv[2])
        x = float(sys.argv[3])
        theta = float(sys.argv[4])
        v0 = float(sys.argv[5])
        m_prime = float(sys.argv[6])
        m0 = float(sys.argv[7])
        mf = float(sys.argv[8])
    except ValueError:
        print("Please provide integer values for all 3 arguments and a float between 0 to 1 for the last one.")
        sys.exit(1)

    # Call the main function with parsed arguments
    main(radius, points, x, theta, v0, m_prime, m0, mf)