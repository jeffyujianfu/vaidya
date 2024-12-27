import numpy as np

# fixed initial conditions
phi0 = 0
theta0 = np.pi/2
r_prime_0 = -1/np.sqrt(2)

# define the mass function
def mass_func(v, slope, m0, mf):
  vf = (mf - m0)/slope
  if v > 0 and v < vf:
    m_prime = slope
    m = m0 + v * m_prime
  elif v > vf:
    m = mf
    m_prime = 0
  else:
    m = 0
    m_prime = 0
  return m, m_prime

# implement the null condition to calcualte v at each time step
def quadroots(a,b,c):
   # calculating discriminant using formula
    dis = b * b - 4 * a * c
    sqrt_val = np.sqrt(abs(dis))

    # checking condition for discriminant
    if a == 0:
      root1 = - c / b
      root2 = root1

    elif dis > 0:
      root1 = (-b + sqrt_val)/(2 * a)
      root2 = (-b - sqrt_val)/(2 * a)

    elif dis == 0:
      root1 = (-b / (2 * a))
      root2 = root1

    # when discriminant is less than 0
    else:
      print("imaginary roots")
      return (1) #returns positive number 1 if roots are imaginary
      root1 = complex(- b / (2 * a), (sqrt_val / (2 * a)))
      root2 = complex(- b / (2 * a), -(sqrt_val / (2 * a)))

    # return the negative root if roots not imaginary, since time is progressing backwards 
    # (we should get a positive and a negative root)
    if dis >= 0:
      if root1 < 0:
        #print(root1,root2)
        return root1
      else:
        #print(root1,root2)
        return root2

# define a function that gets us the value of v_prime_0
def get_v_prime_0(r0, theta0, r_prime_0, theta_prime_0, phi_prime_0, v0, m_prime, m0, mf):
    # determine the mass at observer's time coordinate location
    m = mass_func(v0, m_prime, m0, mf)[0]
    return quadroots(-(1-2*m/(r0)), 2*r_prime_0, ((r0)**2)*((theta_prime_0**2)+((np.sin(theta0))**2)*(phi_prime_0)**2))

# define a function that can get us theta_prime_0 based on a specific looking angle
def get_initial_theta_prime_0 (r_prime_0, r0, theta0):
    return r_prime_0 / r0 * np.tan(theta0)

# the order in the list is v0, r0, theta0, phi0, r_prime_0, theta_prime_0, phi_prime_0, v_prime_0
# Each representing the inital value of the trajectory of light at that coordinate
def initial_conditions(theta_prime_0, phi_prime_0, v0, r0, m_prime, m0, mf):
    return[v0, r0, theta0, phi0, get_v_prime_0(r0, theta0, r_prime_0, theta_prime_0, phi_prime_0, v0, m_prime, m0, mf),
         r_prime_0, theta_prime_0, phi_prime_0]