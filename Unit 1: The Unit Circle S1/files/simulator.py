import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from files.so2 import SO2

class Simulator:
    def __init__(self, controller, commands=[0.0], command_duration=5.0):
        self.controller = controller

        # time
        self.dt = 0.01
        self.t = np.arange(0.0, command_duration*len(commands), self.dt)

        # system parameters
        self.I = 1.0
        self.b = 0.1

        # system state
        self.Phi = SO2()
        self.omega = 0.0

        # commands
        self.theta_c = np.zeros(self.t.shape)
        for i in range(len(commands)):
            self.theta_c[i*self.t.size//len(commands):(i+1)*self.t.size//len(commands)] = commands[i]

        # history
        self.theta_hist = np.zeros(self.t.shape)
        self.omega_hist = np.zeros(self.t.shape)
        self.tau_hist = np.zeros(self.t.shape)

    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', xlim=[-1.2, 1.2], ylim=[-1.2,1.2])
        ax.axis('off')
        ax.set_title(self.controller.name)

        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(th), np.sin(th), 'k-')

        self.command_line = ax.plot([], [], 'ro', label='command')[0]
        self.actual_line = ax.plot([], [], 'bo', label='actual')[0]
        ax.legend(loc='center', numpoints=1)

        return animation.FuncAnimation(fig, self.step, frames=len(self.t), interval=int(1000*self.dt), blit=False, repeat=False)

    def step(self, k):
        # if the controller uses a manifold representation, convert the command to an element of SO(2)
        if self.controller.manifold:
            tau = self.controller.run(SO2.from_angle(self.theta_c[k]), self.Phi, self.omega)
        else:
            tau = self.controller.run(self.theta_c[k], self.Phi.to_angle(), self.omega)

        # propagate dynamics
        self.omega += (-self.b/self.I*self.omega + 1.0/self.I*tau)*self.dt
        self.Phi = self.Phi.dot(SO2.exp(SO2.hat(self.omega*self.dt)))

        # store history
        self.theta_hist[k] = self.Phi.to_angle()
        self.omega_hist[k] = self.omega
        self.tau_hist[k] = tau

        x_c, y_c = SO2.exp(SO2.hat(self.theta_c[k])).visualize()
        x, y = self.Phi.visualize()

        self.command_line.set_data(x_c, y_c)
        self.actual_line.set_data(x, y)

    def plot(self):
        plt.ioff()
        fig = plt.figure()

        ax = fig.add_subplot(311)
        ax.plot(self.t, self.theta_c, 'r-', label='command')
        ax.plot(self.t, self.theta_hist, 'b-', label='actual')
        ax.set_title(self.controller.name)
        ax.set_ylabel('theta (rad)')
        ax.legend()

        ax = fig.add_subplot(312)
        ax.plot(self.t, self.omega_hist, 'b-')
        ax.set_ylabel('omega (rad/s)')

        ax = fig.add_subplot(313)
        ax.plot(self.t, self.tau_hist, 'm-')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('torque (N*m)')

        plt.show()