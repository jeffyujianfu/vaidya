# main.py
from integration import plot_until_xstop, vaidya_geodesic, make_circle, get_preimage_set, get_observer_view
from initialization import mass_func, get_initial_theta_prime_0, initial_conditions, theta0, phi0, r_prime_0
from investigation import find_critical_theta, get_eccentricity, plot_image, impact_parameter_and_lensing_angle
from scipy.interpolate import splprep, splev
from tqdm import tqdm
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# lists fixed initial conditions
phi0 = 0
theta0 = np.pi/2
r_prime_0 = -1/np.sqrt(2)

def main(radius, points, x, theta, v0, m_prime, m0, mf):
    geodesic = vaidya_geodesic
    test = True
    if test:
      test_angle = impact_parameter_and_lensing_angle(geodesic, 0.05, 0.11, m0, mf, test)
    else:
      set_of_impact_parameters = np.linspace(0.00001, 10, 100)
      set_of_rates = [0.01, 0.03, 0.05, 0.0625, 0.0626, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17]
      fig1 = plt.figure(figsize=(15, 8))
      ax1 = fig1.add_subplot(111)
      for rate in set_of_rates:
        angles = []
        set_of_b = []
        for b in set_of_impact_parameters:
          angle = impact_parameter_and_lensing_angle(geodesic, b, rate, m0, mf, test)
          if angle != 0:
            angles.append(angle)
            set_of_b.append(b)
        ax1.plot(set_of_b, angles, label = f"rate = {rate}")
      ax1.grid(True)
      ax1.set_xlabel('Impact parameter')
      ax1.set_ylabel('Lensing angle in degrees')
      ax1.set_title('Gravitational lensing of a single particle')
      ax1.legend()
      fig1.savefig("particle_lensing.png", dpi=300, bbox_inches='tight')
    # # get critical theta
    # theta_c = find_critical_theta(geodesic, radius, x, v0, m_prime, m0, mf)
    # # print(f"Critical theta: {theta_c}")
    
    # # plot eccentricity versus theta
    # theta_list = np.linspace(0.25, np.pi/2.5, 100)
    # eccentricities_theta = []
    # for theta in tqdm(theta_list, desc = "finding eccentricities over theta"):
    #   eccentricities_theta.append(get_eccentricity(geodesic, theta, x, radius, v0, m_prime, m0, mf))
    # fig1 = plt.figure(figsize=(15, 8))
    # ax1.plot(theta_list, eccentricities_theta)
    # ax1.grid(True)
    # ax1.set_xlabel(r'$\theta$')
    # ax1.set_ylabel('Eccentricity of telescope image')
    # ax1.set_title('Eccentricity over shooting angle')
    # # Save fig1
    # fig1.savefig("eccentricity.png", dpi=300, bbox_inches='tight')
    
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
    x_tele_image, y_tele_image = get_observer_view(set_of_trajectories, theta)
    # Parametric interpolation using splprep
    tck, u = splprep([x_tele_image, y_tele_image], s=0, per=True)  # s=0 for smooth interpolation
    u_new = np.linspace(0, 1, 1000)  # Finer parameter space
    x_new, y_new = splev(u_new, tck)  # Interpolated x and y coordinates
    ax3.plot(x_new, y_new)  # Interpolated circle

    # plot the expected circle in 2D
    ax2.scatter(circle[1], circle[2], color = "purple")
    
    fig3 = plt.figure(figsize = (20, 20))
    
    # Add a 3D subplot in the first position of a 1x2 grid
    ax4 = fig3.add_subplot(111)

    # plot trajectories in 3D.
    for xpoints,ypoints,zpoints in set_of_trajectories:
      ax4.plot(xpoints, zpoints, "goldenrod")
      ax4.plot(xpoints[0],zpoints[0], 'ro') # plots red circle at initial point
      ax4.plot(xpoints[-1],zpoints[-1], 'bo') # plots blue circle at end point
      ax4.plot([0], [0], 'o', color='black', markersize=20 * mass_func(v0, m_prime, m0, mf)[0]) # the black hole location
    #ax1.set_ylim3d(-10,10)
    ax4.set_xlim(-x,x)
    #ax1.set_zlim3d(0,170)

    # set titles
    ax4.set_title('Side View of Trajectories')

    # set lables
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    
    # Save fig1
    fig1.savefig("trajectories_3d.png", dpi=300, bbox_inches='tight')
    
    # Save fig2
    fig2.savefig("trajectories_2d.png", dpi=300, bbox_inches='tight')
    
    # Save fig3
    fig3.savefig("trajectories_3d_side_view.png", dpi=300, bbox_inches='tight')
    
    for theta in np.linspace(0.314, 0.523, 8):
      plot_image(radius, points, x, theta, v0, 0.0625, m0, mf, geodesic, 'vaidya_with_naked_singularity', "#3d84bf")
      plot_image(radius, points, x, theta, v0, 0.25, m0, mf, geodesic, 'vaidya_without_naked_singularity', "#255075")
      plot_image(radius, points, x, theta, v0, 0, 1, 1, geodesic, 'schwarzschild', "#0d1d2b")

# system modification for the main function to be called from the command line
if __name__ == '__main__':
    # Check if the correct number of arguments was provided
    if len(sys.argv) != 9:
        print("Usage: python3 main.py <radius> <points> <x> <theta> <v0> <m_prime> <m0> <mf> <eccentricity_plot>")
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