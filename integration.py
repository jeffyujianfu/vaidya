import numpy as np
import scipy as sp #for numerical ODE solver
import matplotlib.pyplot as plt #for visualization
from mpl_toolkits.mplot3d import Axes3D
from initialization import initial_conditions, mass_func
from tqdm import tqdm
np.set_printoptions(precision=4)

from functools import partial

# step 1: define the vaidya geodesic
def vaidya_geodesic(t, z, slope, m0, mf):
  # define z as array of the variables, are all functions of t (tau)
  v, r, theta, phi, a, b, c, d = z
  # define mass and mass_prime
  m, m_prime = mass_func(v, slope, m0, mf)
  # equation for v'
  z0 = a
  # equation for r'
  z1 = b
  # equation for θ'
  z2 = c
  # equation for φ'
  z3 = d
  # equation for a' which is v"
  z4 = (-(m)/(r**2))*(a**2)+r*((np.sin(theta))**2)*(d**2) + r*(c**2)
  # equation for b' which is r"
  z5 = ((m_prime/r)+((m*(r-2*m))/(r**3)))*(-1)*(a**2) + (r-2*m)*(((np.sin(theta))**2)*(d**2)+c**2) + 2*(m/(r**2))*b*a
  # equation for c' which is θ"
  z6 = (np.sin(theta))*(np.cos(theta))*(d**2) - (2/r)*b*c
  # equation for d' which is φ"
  z7 = -2*((np.cos(theta))/(np.sin(theta)))*d*c - (2/r)*d*b
  return[z0,z1,z2,z3,z4,z5,z6,z7]


# step 2: define the circle to shoot for
def make_circle(r, h, x, points):
  #initialize arrays of 0s length of points
  z = [0] * points
  y = [0] * points
  x_list = [x] * points
  #put in initial point
  z[0] = h
  y[0] = 0

  #initialize theta to be at the top of circle
  theta = np.pi/2
  delta_theta = 2*np.pi/points
  #go around, finding point on circle 2pi/(points) radians clockwise of last
  for i in range(points - 1):
    theta = theta - delta_theta
    yi = r * np.cos(theta)
    y[i+1] = yi
    zi = r * np.sin(theta) + h - r
    z[i+1] = zi

  return (x_list,y,z)


# step 3: perform integration of the vaidya geodesic
# Define the event function to stop when x crosses desired_xval
def event_x_reached(t, y, slope, m0, mf, desired_xval = -40):
    r = y[1]
    theta = y[2]
    phi = y[3]
    xpoint = r * np.sin(theta) * np.cos(phi)
    return xpoint - desired_xval

# Set the event function properties
event_x_reached.terminal = True  # Stop the integration when the event is reached
event_x_reached.direction = 0     # Detect zero-crossings in any direction

# Define the event function to stop when location of the photon falls into the apparent horizon
def event_sphere(t, y, slope, m0, mf):
    v = y[0]   # Assuming v is the first component of the state vector
    r = y[1]
    current_mass = mass_func(v, slope, m0, mf)[0]
    return r - 3 * current_mass

event_sphere.terminal = True  # Stop the integration
event_sphere.direction = -1    # Trigger when crossing from positive to negative

# Define the event function to stop whenever a numerical failure occurs
def event_numerical_failure(t, y, slope, m0, mf):
    # Check for NaN/inf in the state vector `y`
    if np.isnan(y).any() or np.isinf(y).any():
        return 0.0  # Trigger event if invalid
    
    # Optional: Check for unphysical values (e.g., negative radius)
    r = y[1]
    if r <= 0:
        return 0.0
    
    return 1.0  # No issue

event_numerical_failure.terminal = True  # Terminate integration
event_numerical_failure.direction = -1   # Trigger when crossing from 1 → 0

# solve the geodesic until the photon reaches the desired x value
def plot_until_xstop(geodesic, initials, desired_xval, slope, m0, mf, check = False):
  sol = sp.integrate.solve_ivp(geodesic,
                              [0,1000],
                              initials,
                              args=(slope, m0, mf),
                              rtol=1e-20,
                              atol=1e-20,
                              events=[event_x_reached, event_sphere, event_numerical_failure])
  # Check which event occurred
  if check: 
    if len(sol.t_events[0]) > 0:
      print("Stopped because x reached desired_xval")
    elif len(sol.t_events[1]) > 0:
      print("Stopped because x² + y² < 4m²")
  rpoints = sol.y[1]
  thetapoints = sol.y[2]
  phipoints = sol.y[3]
  # convert spherical to cartesian to plot
  xpoints = [0] * len(rpoints)
  ypoints = [0] * len(rpoints)
  zpoints = [0] * len(rpoints)
  for i in range(len(rpoints)):
    zpoints[i] = (rpoints[i]) * (np.cos(thetapoints[i]))
    ypoints[i] = (rpoints[i]) * (np.sin(thetapoints[i])) * (np.sin(phipoints[i]))
    xpoints[i] = (rpoints[i]) * (np.sin(thetapoints[i])) * (np.cos(phipoints[i]))

  #---------------------------------------------------------------------
  #interpolation to get value between x just below and just above target
  #to reduce error, assumes linearity of trajectory close to point

  x1 = xpoints[-1]
  x0 = xpoints[-2]
  t = (desired_xval - x0)/(x1 - x0)
  xpoints[-1] = (1-t) * x0 + t * x1
  y1 = ypoints[-1]
  y0 = ypoints[-2]
  z1 = zpoints[-1]
  z0 = zpoints[-2]
  ypoints[-1] = (1-t) * y0 + t * y1
  zpoints[-1] = (1-t) * z0 + t * z1
  #---------------------------------------------------------------------
  return ([xpoints,ypoints,zpoints])


# step 4: find the initial conditions for the shooting problem
# define a function that can get us to the preimage of a point using step algorithm
def get_preimage (geodesic, new_point, current_point, current_initial_condition,
                  v0, r0, m_prime, m0, mf, error = 0.01):
  guess_point = current_point
  new_initial_conditions = current_initial_condition
  theta_prime_0 = current_initial_condition[6]
  phi_prime_0 = current_initial_condition[7]
  delta_theta_prime = 10**(-5)
  delta_phi_prime = 10**(-5)

  while np.abs(new_point[1] - guess_point[1]) + np.abs(new_point[2] - guess_point[2]) > 2*error:
    # adjusting values for delta phi prime
    if np.abs(new_point[1] - guess_point[1]) > 1:
      delta_phi_prime = 10**(-4)
    elif np.abs(new_point[1] - guess_point[1]) > 0.1:
      delta_phi_prime = 10**(-5)
    else:
      delta_phi_prime = 10**(-6)

    # adjusting values for delta theta prime
    if np.abs(new_point[2] - guess_point[2]) > 1:
      delta_theta_prime = 10**(-4)
    elif np.abs(new_point[2] - guess_point[2]) > 0.1:
      delta_theta_prime = 10**(-5)
    else:
      delta_theta_prime = 10**(-6)

    # update theta_prime_0 and phi_prime_0 accordingly
    if new_point[1] > guess_point[1]:
      phi_prime_0 = phi_prime_0 + delta_phi_prime
    else:
      phi_prime_0 = phi_prime_0 - delta_phi_prime
    if new_point[2] > guess_point[2]:
      theta_prime_0 = theta_prime_0 - delta_theta_prime
    else:
      theta_prime_0 = theta_prime_0 + delta_theta_prime
    initial_angles = [theta_prime_0, phi_prime_0]
    new_initial_conditions = initial_conditions(*initial_angles, v0, r0, m_prime, m0, mf)
    trajectory = plot_until_xstop(geodesic, new_initial_conditions, -40, m_prime, m0, mf)
    guess_point = [trajectory[0][-1], trajectory[1][-1], trajectory[2][-1]]
    #print("delta vertical", np.abs(new_point[2] - guess_point[2]), "away from the target point.")
    #print("delta horizontal", np.abs(new_point[1] - guess_point[1]), "away from the target point.")
  return new_initial_conditions


# get the preimage of a set of points by looping through the preimage function
def get_preimage_set (geodesic, set_of_points, current_point, current_initial_condition, 
                      v0, r0, m_prime, m0, mf):
  set_of_points.pop(0)
  set_of_preimages = []
  loop = set_of_points
  # Only use tqdm if the number of points is large enough
  if len(set_of_points) > 4:
      # For larger loops, enable tqdm
      loop = tqdm(set_of_points, desc="Finding preimage")
  for point in loop:
      set_of_preimages.append(get_preimage(geodesic, point, current_point, current_initial_condition,
                                           v0, r0, m_prime, m0, mf))
  return set_of_preimages


# Interpolate between the first and second point along the trajectory
# Where we want the xval to stop at 10**(-9) away from x0.
# Then we project the vertical view onto the slanted plane
def get_observer_view(set_of_trajectories, theta):
  d = 10**(-9) # distance from x0 to stop at
  xstuff = []
  ystuff  =  []
  zstuff = []

  for trajectory in set_of_trajectories:
    # interpolate
    #--------------
    x1 = trajectory[0][1]
    x0 = trajectory[0][0]
    t = - d/ (x1 - x0)
    xstuff.append((1-t) * x0 + t * x1)
    y1 = trajectory[1][1]
    y0 = trajectory[1][0]
    z1 = trajectory[2][1]
    z0 = trajectory[2][0]
    ystuff.append((1-t) * y0 + t * y1)
    zstuff.append((1-t) * z0 + t * z1)
  
  # change them into arrays
  ystuff = np.array(ystuff)
  zstuff = np.array(zstuff)
  
  # project onto slanted plane
  z_max = max(zstuff)
  ystuff = ystuff * (z_max**2 + d**2) / (d**2 + z_max * zstuff)
  zstuff = z_max/np.cos(theta) - np.sqrt(z_max**2 + d**2) * d * (z_max - zstuff) / (d**2 + z_max * zstuff)
  return (ystuff, zstuff)