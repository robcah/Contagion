"""
Animation simulating a contagion of an infectious disease
by Roberto Cahuantzi, inspired by the article of 
the Washingtong Post:
www.washingtonpost.com/graphics/2020/world/corona-simulator/
based on the code:
"Elastic collisions with Gravity" by Jake Vanderplas
(vanderplas@astro.washington.edu,
http://jakevdp.github.com
license: BSD)
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import matplotlib as mpl

class Population:
    """Orbits class
    
    init_state is an [N x 4] array, where N is the number of individuals
        the frst two values are the x, y coordinates, while the last 
        two are the velocities in both axes:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]
    
    sick is the original infection condition of the population
    
    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    
    size define the contact perimeter
    
    M stands for mass
    
    fps is the frames per second
    
    N is number of undividuals in the population
    
    qrtn is the fraction of the population in quarantine 
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 sick = [0, 1, 0],
                 bounds = [-3, 3, -2, 2],
                 size = 0.04,
                 M = 0.05,
                 fps = 30,
                 N = 200,
                 qrtn = 0.5):
        
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.sick = sick
        self.immune = np.zeros_like(sick)
        self.sick_t = np.zeros_like(sick)
        self.colors = sick * 0.5

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # define immobile one's (in quarantine) indices
        imb = int(N*qrtn)        
        
        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of undividuals undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 4 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass in case of quarantine they are immobile
            if i1 <= imb:
                m1 = 1e6 # just big value
            else:
                m1 = self.M[i1]
            if i2 <= imb:
                m2 = 1e6 # just big value
            else:
                m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2)
            
            # define new condition
            s1 = self.sick[i1]
            s2 = self.sick[i2]
            im1 = self.immune[i1]
            im2 = self.immune[i2]
                        
            if s1 == 1 and s2 == 0 and im2 != 1:
                self.sick[i2] = 1
            
            if s1 == 0 and s2 == 1 and im1 != 1:
                self.sick[i1] = 1
            
            # counting of illnes time
            self.sick_t[i1] += 1 if s1 == 1 else 0
            self.sick_t[i2] += 1 if s2 == 1 else 0
            
            # define condition of immunity
            st1 = self.sick_t[i1]
            st2 = self.sick_t[i2]
            
            sec_sick = 2.5 # seconds ill to reach immunity
            if st1 >= sec_sick * fps:
                self.sick[i1] = 0
                self.immune[i1] = 1
                
            if st2 >= sec_sick * fps:
                self.sick[i2] = 0
                self.immune[i2] = 1
            
        # assigning color code
        self.colors = np.array([0.5 if s == 1 else 
                           1.0 if (s == 0 and im == 1) else 
                           0.0 for s, im in zip(
                            self.sick[:], self.immune[:])])

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1
        
    
            


#------------------------------------------------------------
# set up initial state
N = 200 # Number of individuals in the population
fps = 30 # frames per second of the video
frames = 20 * fps # length of the video (seconds * fps)
inf_0 = 1 # number of initial infected
size = 3.0 # size of the individual
cmap = mpl.cm.get_cmap('brg') # colormap used
qrtn = 0.25 # fraction in quarantine
# np.random.seed(0) # decomment for reproducible results

# assignment of the initial position, movement and health
init_state = -0.5 + np.random.random((N, 4))
init_state[:, 0] *= 5.9
init_state[:, 1] *= 3.9
init_state[:int(N *qrtn), 2:] = 0.0
init_state[:, 2:] *= 2.0
sick_state = np.zeros(N)
inf_rand = np.random.randint(0, N, inf_0)
sick_state[inf_rand] = 1

# timeseries of infection and immunisation
frame = np.arange(0, frames)
share_sick = np.zeros((frames, 2))
share_sick[0, 0] = sum(sick_state[:]) / N

# definition of the class
box = Population(init_state, size=size/100, sick=sick_state, 
                  fps = fps, N = N, qrtn = qrtn)
dt = 1. / fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
grid = plt.GridSpec(4, 4, hspace=-0.1, wspace=-0.2, 
                    top=0.945, bottom = -0.05,
                    right = 0.95, left = 0.05)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(grid[1:, :], aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))
ax_pdf = fig.add_subplot(grid[0, 1:3], xticklabels=[], 
                         xlim=(0, frames), ylim = (0, 1))
plt.title(f'N = {N}, {qrtn * 100:.1f}% in quarantine')
ax.set_axis_off()

# undividuals holds the locations
undividuals = ax.scatter(init_state[:, 0], init_state[:, 1],
                       c = box.colors[:], s = size*3, 
                       cmap = cmap, vmin = 0.0, vmax = 1.0)

share = ax_pdf.fill_between(frame, share_sick[:, 0], 
                            facecolor = cmap(0.0))

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect, pdf_ax, ax
    undividuals.set_offsets(init_state[:, :2])
    rect.set_edgecolor('none')
#     share.get_paths()
    return undividuals, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig, frame, share_sick, N, share
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    # individuals
    rect.set_edgecolor('k')
    undividuals.set_offsets(box.state[:, :2])
    undividuals.set_array(box.colors[:])
    
    # density chart
    sick = box.sick[:]
    imm = box.immune[:]
    share_sick[i, 0] = sum(sick) / N
    share_sick[i, 1] = sum(imm) / N
    sick_top = share_sick[:, 0]
    imm_top = share_sick[:, 0] + share_sick[:, 1]
    ax_pdf.collections.clear()
    share = ax_pdf.fill_between(frame, sick_top, 
                                facecolor = cmap(0.5))
    ax_pdf.fill_between(frame, sick_top, imm_top, 
                                facecolor = cmap(1.0))
    ax_pdf.fill_between(frame, imm_top, np.ones_like(frame), 
                                facecolor = cmap(0.0))
    
    return undividuals, rect, share

# creating animation
ani = animation.FuncAnimation(fig, animate, frames=frames,
                              interval=10, blit=True)#, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

#saving animation
ani.save(f'Contagion_N{N}_Q{qrtn*100:.0f}%.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
plt.show()