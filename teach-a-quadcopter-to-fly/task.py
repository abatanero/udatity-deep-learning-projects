import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, task_type=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Initial position
        self.init_pos = init_pose
        
        # Current position
        self.previous_pos = init_pose
        
        # Task type
        if task_type in ('take-off', 'fly-to-target'):
            self.task_type = task_type
        else:
            self.task_type = 'fly-to-target'
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, done=False):
        """Uses current pose of sim to return reward."""
        if self.task_type == 'take-off':
            # For the take off task we only want the agent to move the z direction, therefore, movement in other directions is penalized
            current_position = self.sim.pose[:3]
            mov_x = (self.target_pos[0] - current_position[0]) / (self.target_pos[0] - self.init_pos[0] + 1e-1)
            mov_y = (self.target_pos[1] - current_position[1]) / (self.target_pos[1] - self.init_pos[1] + 1e-1)
            mov_z = (self.target_pos[2] - current_position[2]) / (self.target_pos[2] - self.init_pos[2] + 1e-1)

            # extra reward for continuing flying, penalized by the relative distance to the target, with stronger factor on the z direction
            reward = self.target_pos[2]*10 + 0.0001 * (-1 * mov_x**2  - 1 * mov_y**2 - 10 * mov_z**2)
            if self.previous_pos[2] > current_position[2]:
                reward = -reward
            # if done and np.sum((self.target_pos - current_position)**2) < 2:
            #     reward += 5000 * (self.runtime - self.time) / self.dt
        
        elif self.task_type == 'fly-to-target':
            # For the fly to target task we only want the agent to move in any direction
            reward = 1.-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        self.previous_pos = self.sim.pose
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        
        # Modify the done variable to True in case the quadcopter passed close enough to the target
        if self.task_type in ('take-off') and np.sum((self.target_pos - self.sim.pose[0:3])**2) < 1:
            done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        self.previous_pos = self.init_pos
        return state