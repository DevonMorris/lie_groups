import numpy as np


class SO2:
    def __init__(self, G=np.eye(2)):
        self.mat = G

    def dot(self, rhs):
        return SO2(np.dot(self.mat, rhs.mat))

    @staticmethod
    def identity():
        return SO2(np.eye(2))

    def inverse(self):
        return SO2(self.mat.T)

    @staticmethod
    def exp(g):
        omega = SO2.vee(g)
        return SO2(np.array([[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]]))

    def log(self):
        theta = self.to_angle()
        return np.array([[0, -theta], [theta, 0]])

    @staticmethod
    def hat(omega):
        return np.array([[0.0, -omega],[omega, 0.0]])

    @staticmethod
    def vee(g):
        return g[1,0]

    def to_angle(self):
        return np.arctan2(self.mat[1,0], self.mat[0,0])

    @staticmethod
    def from_angle(theta):
        return SO2.exp(SO2.hat(theta))

    def visualize(self):
        pos = self.mat.dot(np.array([1.0, 0]))
        return pos[0], pos[1]

    def __str__(self):
        return self.mat.__str__()
    
    def __repr__(self):
        return self.mat.__str__()