import numpy as np
import matplotlib.pyplot as plt
import pickle

class state_vector:
    # Define __init__ method that initializes the object with the state vector, time vector, name of the object, and name of the parameters
    def __init__(self, al = [0], t = [0], name = 'uninitialized', str_figures_path = 'unknown', paraname = 'unknown'):
        self.tdim = np.size(t)
        self.al = al
        self.aldim = np.size(al)
        self.statedim = np.size(al[0:3])
        self.pdim = np.size(al[3:3])
        self.x0 = al[0:3]
        self.p0 = al[3:3]
        self.t = t
        self.name = name
        self.figures_path = str_figures_path
        self.trajectory = np.zeros([self.tdim, self.aldim])
        self.Enstrajectory = np.zeros([self.tdim, self.aldim, 7])
        self.Xatrajectory = np.zeros([self.tdim,self.aldim,7])
        self.paraname = paraname
        print("State_Vector Object Created!")
        print("State_Vector Object Name: ", self.name)
        print("al: ", self.al)
        print("t: ", self.t)
        print("x0: ", self.x0)
 

    # Define __str__ method that prints the name of the object and the number of states and parameters when the object is printed (e.g., print(object))    
    def __str__(self): 
        print(self.name)
        print('Number of states and parameters')
        print(self.aldim)
        print('Parameters:')
        print(self.p0)
        print('Number of parameters:')
        print(self.pdim)
        print('Initial condition:')
        print(self.x0)
        print('Number of states:')
        print(self.statedim)
        print('Trajectory:')
        print(self.trajectory)
        print('EnsTrajectory:')
        print(self.Enstrajectory)
        print('Figure Path:')
        print(self.figures_path)
        return f"State vector object:\n - name: {self.name}\n - time steps: {self.t}\n - figures path: {self.figures_path}"
    
    
    # Define setFigurePath method that sets the path for saving figures:
    def setFigurePath(self, str_figures_path):
        self.figures_path = str_figures_path


    # Define setTrajectory method that sets the trajectory of the object:
    def setTrajectory(self,states):
        self.trajectory = states
    
    
    # Define getTrajectory method that returns the trajectory of the object:
    def getTrajectory(self):
        return self.trajectory
    
    
    # Define times method that returns the time vector of the object:
    def getTimes(self):
        return self.t
    
    
    # Define save method that saves the object to a file:
    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self, output)
    
    
    # Define load method that loads the object from a file:
    def load(self, infile):
        with open(infile, 'rb') as input:
            sv = pickle.load(input)
        return sv
    
    
    # Plot Trajectory:
    def plotTrajectory(self, states, title, outfile):
        fig = plt.figure(figsize = (14,6))
        ax  = fig.add_subplot(111)
        plt.plot(states[0, 0:5000], linestyle = 'dashed', marker='.', markersize = 2, color = 'blue')
        plt.axhline(y=0, linestyle='dashed', color = 'grey')
        plt.axvline(x=0, linestyle='dashed', color = 'red')
        plt.xlabel('Time Step (1 step = 0.01)', fontsize = 16)
        plt.ylabel(f'State $x$', fontsize = 16)
        plt.title(title, fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.grid(color = 'grey', linestyle = ':', linewidth = 0.5)
        plt.show()
        fig.savefig(self.figures_path + outfile)
        plt.close()

