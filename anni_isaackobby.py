# Python 3.6
#
# Guidelines:
#
#   - It is not allowed to import additional packages/modules.
#
#   - It is not allowed to modify how the requested functions are
#   invoked. This means that the list of input arguments a function
#   takes, or the list of output variables a function returns should
#   be the same as in this template. Otherwise, your functions will
#   not work when we run the test code.
#
#   - Please be aware of the shapes of the vectors and matrices. A
#   column vector of length N is represented here by a numpy.ndarray
#   of shape (N,). Refer to the help in each function header.

import numpy as np

#############################################################################
# Section 1. Requested functions
#   - Please code the requested functions in this section. You can
#   paste any auxiliary function that you want to write in Section 2.
#   - Please do not erase the functions that you decided not to implement.
#############################################################################


class Functions:
    def equilateral(x, y):
        """
        Description: see problem statement.

        Args:
            `x` and `y` are np.ndarrays of shape (N,)

        Returns: 
            `z` is an np.ndarray of shape (N,)
        """

        # Replace with your code
#         for i in range(len(x)):
            
#             x = np.array((np.random.choice(x), np.random.choice(y))) # pick the first point 
#             y = np.array((np.random.choice(x), np.random.choice(y)))  # pick the second point
        
        x_cord = ( x[0]+ y[0] + np.sqrt(3)*(x[1]-y[1]) )/2 # get the `x_coordinate` of the third vertex
    # print(f"this is x_cord {x_cord}")

        y_cord = ( x[1]+ y[1] + np.sqrt(3)*(y[0]-x[0]) )/2 # get the `y_coordinate` of the third vertex
    # print(f"this is y_cord {y_cord}")


        v_3 = np.array([x_cord, y_cord]) # get the third vertex
    # print(f"V_3 is like this before the conditional statement{v_3}")
        if equilateral_condition(x, y, v_3) != 'None':
            return v_3
        else:
            return None
#             break
#         else:
#             continue

    def nearest_on_circle(c, R, x):
        """
        Description: see problem statement.

        Args:
            `c` and `x` are np.ndarrays with shape (2,)
            `R` is of a numeric type (e.g. float)

        Returns: 
            `y` is an np.ndarray of shape (2,)
        """

        # Replace with your code
        epsilon = 0.01 * R # assuming point y is 10% closest to x, if all points lie on the radius of the circle, thus, c, y, x.
        y_cord = (1-epsilon) * c + epsilon * x # compute the cordinates  of y
        return y_cord #np.array([1., 1.])

    def reflections_in_tunnel(w, h, alpha, n):
        """
        Description: see problem statement.

        Args:
            `w`, `h`, `alpha`, `n` are of a numeric type (e.g. float)

        Returns: 
            `x_n` is an np.ndarray with shape (2,)

        """

        # Replace with your code
        x_n = np.zeros(2) # set the final position to zeros

        x_0 = np.array([0, h]) # initial coordinate

        # update the position
        x_n[0] = x_0[0]
        x_n[1] = x_0[1]

        # Compute the incidence angle
        angle_i = 180 - (alpha + 90) # incidence ray makes a right angle-trianle
        angle_i = angle_i * np.pi/180 # convert degrees into randian
        alpha = alpha * np.pi/180

        # compute the distance in the reflection ray along the x-axis
        d1_on_axis = ( (w-h) * np.sin(angle_i) )/ np.sin(alpha) # the reflected laser makes 90 degrees with the width which helps to compute the next movement

        # update new position
        x_n[0] = 0 + d1_on_axis # get the x-cordinate in the next direction
        x_n[1] = w # get the y-cordinate in the next direction

        # since the distance computation is different from the 
        # initial occurence, I use for loop for the uniform reflections
        for i in range(2, n+1): # this takes care of the initial coordinates
            d_i_on_axis = (w * np.sin(angle_i) ) / np.sin(alpha) # compute new distance
            if i%2 == 0: # downward reflection
                x_n[0] = x_n[0] + d_i_on_axis # update the x-coordinate
                x_n[1] = 0 # update the y-coordinate back to zero
            else: # upward reflection
                x_n[0] = x_n[0] + d_i_on_axis
                x_n[1] = w # y_cordinate is always at the height of the width.
        return  x_n 

    def reflection_on_circle(c_x, R, alpha):
        """
        Description: see problem statement.

        Args:
            `c_x`, `R`, `alpha` are of a numeric type (e.g. float)

        Returns: 
            `L` is of a numeric type (e.g. float)

        """

        # Replace with your code
        i_angle = alpha + int(0.10*alpha) # Since incidence angle is not, a simple analogy is a 10% change in angle of source beam to its incidence
        r_angle = i_angle
        angle_at_L = 180 - ((2 * i_angle) + alpha)
        l = ( (R * np.sin(r_angle)) / np.sin(angle_at_L) ) + c_x
        return l

    def repeated_entries(x):
        """
        Description: see problem statement.

        Args:
            `x` is an np.ndarray of shape (N,)

        Returns: 
            `S` is an np.ndarray of shape (N,L)
            `u` is an np.ndarray of shape (L,)

        """

        # Replace with your code
        N = len(x) # Get the length of x
        u = np.array([i for i in set(x)]) # get (u)_i < (u)_j
        L = len(u)
        u_ = u.reshape(-1, 1) # reshape `u` such that `ndim` >= 2 for inverse calculation
        # To find `S` , I have to solve the matrix equation `u^-1 * x`.
        u_inverse = np.linalg.inv(u_.T @ u_) @ u_.T
        S = x.reshape(-1, 1) @ u_inverse
        return S, u

    def orthogonal_components(B, v):
        """
        Description: see problem statement.

        Args:
            `B` is an np.ndarray of shape (M, N)
            `v` is an np.ndarray of shape (M, )

        Returns: 
            `v_par` and `v_perp` are np.ndarrays of shape (M, )

        """
#         print(f"shape of v is {v.shape}")
        # Replace with your code
        v_perp = orth_vector(B)
#         print(f"shape of v_perp is {v_perp.shape}")
        v_par = v - v_perp
        return v_par, v_perp

    def orthogonal_vector(B):
        """
        Description: see problem statement.

        Args:
            `B` is an np.ndarray of shape (M, N)

        Returns: 
            `v` is an np.ndarray of shape (M, )

        """

        # Replace with your code
        r_vec = np.random.rand(B.shape[0], 1) # Generate random vectors
        B_stack = np.hstack((B, r_vec)) # add additional random vector temporarily
        b = np.zeros(B.shape[1] + 1) # `b` zeros for orthogonal vectors
        b[-1] = 1 # Set the value for random vector to be non zero
        v = np.linalg.lstsq(B_stack.T, b, rcond=None)[0] # Solve for `x` using A @ x = b 
        return v

    def marginal_x(P):
        """
        Description: see problem statement.

        Args:
            `P` is an np.ndarray of shape (M, N)

        Returns: 
            `p_x` is an np.ndarray of shape (M, )

        """

        # Replace with your code
        p_x = [np.sum(P[i]) for i in range(len(P))] # compute the marginal probability along the rows, thus, x.
        p_x = np.array(p_x).T # turn list into ndarray and transpose as required. 
        return p_x 

    def expectation_y(P):
        """
        Description: see problem statement.

        Args:
            `P` is an np.ndarray of shape (M, N)

        Returns: 
            `e` is of a numeric type (e.g. float)

        """

        # Replace with your code
        e_ = [(i * np.sum(P[:, i])) for i in range(P.shape[1])]
        e = np.sum(np.array(e_))
        return e

    def conditional_x(P, n):
        """
        Description: see problem statement.

        Args:
            `P` is an np.ndarray of shape (M, N)
            `n` is an integer between 1 and N.

        Returns: 
            `p_x_y` is an np.ndarray of shape (M, )

        """

        # Replace with your code
#         marg_y = np.sum(P[:, n]) # marginal probability at y = n
#         cond_x = [prob_dist / marg_y for prob_dist in P[:, n]] # conditional probabilities for all x given y = n
#         p_x_y = np.array(cond_x) # convert list into np.ndarray
        p_x_y = conditional_prob_x(P, n)
        return p_x_y

    def conditional_expectation_x(P, n):
        """
        Description: see problem statement.

        Args:
            `P` is an np.ndarray of shape (M, N)
            `n` is an integer between 1 and N.

        Returns: 
            `e` is of a numeric type (e.g. float)

        """

        # Replace with your code
        p_x_y = conditional_prob_x(P, n) # Get the conditional probabilities with the help of the conditional_x method.
        cond_prob = [prob * obser for prob, obser in zip(p_x_y, range(len(p_x_y)))] # compute expectation for all observations
        cond_prob = np.array(cond_prob) # convert list into ndarray
        e = np.sum(cond_prob) # compute the mean of the expectations
        return e


#############################################################################
# Section 2. Auxiliary functions
#   - Please place here the additional functions that you would like
#   to write to assist the functions in Section 1.
#############################################################################

# Helper function to check for the condition of an equilateral triangle, the `x, y` given in the 
# checks.py function has length of N>2 which comes with a challenge for a vertex of a triangle.
# I supposed that the x, y given with length ==5 could be any values in a point (x1,y1), (x2, y2) chosen at random
# and equilateral rules binding a tringle applied to obtained a third vertex `z`, (x3, y3), that satisfies the equal norm condition.
# Below function performs the above explained checks, however, it doesn't work for the provided checks.py metric because of mixmatch shapes. 
def equilateral_condition(x, y, z):
    norm_xz = np.round(np.linalg.norm(x-z), 8) 
    norm_yz = np.round(np.linalg.norm(y-z), 8)
    norm_xy = np.round(np.linalg.norm(x-y), 8)
    if norm_xz == norm_yz == norm_xy:
        return z
    else:
        return "None"

    
    # A helper that solves for `x` by the least square method of `A@x = b`
def orth_vector(B):
        """
        Description: see problem statement.

        Args:
            `B` is an np.ndarray of shape (M, N)

        Returns: 
            `v` is an np.ndarray of shape (M, )

        """

        # Replace with your code
        r_vec = np.random.rand(B.shape[0], 1) # Generate random vectors
        B_stack = np.hstack((B, r_vec)) # add additional random vector temporarily
        b = np.zeros(B.shape[1] + 1) # `b` zeros for orthogonal vectors
        b[-1] = 1 # Set the value for random vector to be non zero
        v = np.linalg.lstsq(B_stack.T, b, rcond=None)[0] # Solve for `x` using A @ x = b 
        return v

    # a helper for conditional probabilities computation for joint distribution
def conditional_prob_x(P, n):
        """
        Description: see problem statement.

        Args:
            `P` is an np.ndarray of shape (M, N)
            `n` is an integer between 1 and N.

        Returns: 
            `p_x_y` is an np.ndarray of shape (M, )

        """

        # Replace with your code
        if P.shape[1]<=1:
            marg_y = np.sum(P[:]) # marginal probability at y = n
            cond_x = [prob_dist / marg_y for prob_dist in P[:]] # conditional probabilities for all x given that P is of shape m*n, where n=1
        else:
            marg_y = np.sum(P[:, n-1])
            cond_x = [prob_dist / marg_y for prob_dist in P[:, n-1]] # conditional probabilities for all x given y = n  
        p_x_y = np.array(cond_x) # convert list into np.ndarray
        return p_x_y
   
def softMax(array):
    scores = []
    for i in range(len(array)):
        scores.append(np.exp(array[i]) / np.sum(np.array([np.exp(j) for j in array])))
    return scores

#####################################################
 # My Concerns
#####################################################
# 1. First of all I will like to thank you for this opportunity to being shortlisted.
# 2. I will like to mention that I like the problems for the background test.
# 3. Unfortunately, I have to work from the hospital at the moment because of issue with illness I happen to battle with  wihtinn the timeframe of the background test. I am getting better and I will soon be.
# 4. My concerns were with the question 1a, 2a and 2b. 
# for 1a, I do not quite understand the nature of the inputs  of x and y. They are five values in a vector format. Reading the question I thought they should have been vertices and that z also have be a vertex. I do not understand why?

# also, for 2a, my function does returns S and u as required but I do not understand the manner with which it is assessed.

# Lastly, for the 2b, the assessment comes out as 'fail', however, my 'v_perp' is perpendicular to the given matrix B and with a simple algebraic-matrix operation according to the given operations 'v_par' should be computed by change of subject. However it is not so.

# I believe I have the technical know-how to handle these problems, I would have done better with a normal condition from home unlike here at the hospital. Thank you.
