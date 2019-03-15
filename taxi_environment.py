import numpy as np

class environment():
    '''
    The taxi enviroment is defined here
    here are the possible actions and the places
    where the taxi is allowed to go
    '''

    def __init__ (self, map_size, map, debug = False):
        self.map_size = map_size
        self.map = map
        self.current_taxi_position = (0,0)
        self.current_passenger_position = 0
        self.current_passenger_destination = 0
        self.positions_names = ['R','G','Y','B','Taxi']
        self.position_index = {'R': 0 ,
                               'G': 1,
                               'Y': 2,
                               'B': 3,
                               'Taxi': 4}
        self.positions = {'R': (0,0),
						  'G': (map_size[1] - 1, 0),
						  'Y': (0, map_size[0] - 1),
						  'B': (map_size[0] - 1, map_size[0] - 2)}

        self.actions_names = ['up','down','left','right','pickup','dropoff']
        self.actions = {	'up': np.array([-1,0]),
							 'down': np.array([1,0]),
							'right': np.array([0,1]),
							 'left': np.array([0,-1]),
							'pickup': 0,
							'dropoff': 0}
                
        self.debug = debug

    def initialize_locations(self):
        '''
        Initialize randomly taxi and passengers position
        '''
        self._init_passenger_loc()
        self._init_taxi_loc()
        if (self.debug) :
            self.plot_map()
        return

    def _init_passenger_loc(self):
        starting_points = self.positions_names[:-1]
        random_starting_point = np.random.choice( starting_points )
        self.current_passenger_position = self.position_index [random_starting_point] 
        
        #starting_points.remove( random_starting_point )
        random_starting_point = np.random.choice( starting_points )
        self.current_passenger_destination = self.position_index [random_starting_point]
        return

    def _init_taxi_loc(self):
        x_loc = np.random.choice( np.arange ( self.map_size[0] ) )
        y_loc = np.random.choice( np.arange ( self.map_size[0] ) )
        self.current_taxi_position = ( x_loc, y_loc )
        return
 
    def taxi_move(self, direction):
        '''
        move taxi with
        up, down, right or left
        actions
        '''
        self.map
        taxi_position_in_map = np.array([self.current_taxi_position[0] + 1, 2*self.current_taxi_position[1] + 1])
        new_direction_in_map = taxi_position_in_map + direction
        if ((self.map[new_direction_in_map[0]][new_direction_in_map[1]] != '|') and self.map[new_direction_in_map[0]][new_direction_in_map[1]] != '-' ):
            new_direction = self.current_taxi_position + direction
        else:
            new_direction = self.current_taxi_position
        self.current_taxi_position = new_direction
        return

    def check_passenger_in_taxi(self):
        if (self.positions_names [self.current_passenger_position] == 'Taxi' ):
            return True
        else:
            return False

    def check_passenger_dropoff(self):
        loc_dest = self.positions [ self.positions_names [self.current_passenger_destination] ]

        if (self.current_taxi_position == loc_dest):
            return True
        else:
            return False
       
    def perform_action(self, action):
        '''
        Perform action 1 to 6
        receive the reward accordingly
        '''
        reward = -1
        finish_episode = False 

        # if a moving action
        if (action <= 3):
            direction_name = self.actions_names[action]
            direction = self.actions[direction_name]
            self.taxi_move(direction)
            if (self.debug) :
                print()
                print(self.actions_names[action])
        # if pickup action
        elif ( action == 4 ):
            if (self.debug) :
                print()
                print(self.actions_names[action])
            if (self.check_passenger_in_taxi()):
                reward += -10
            elif ( np.array_equal ( self.positions [self.positions_names[ self.current_passenger_position] ], self.current_taxi_position) ):
                self.current_passenger_position = 4
                if (self.debug) :
                    print('passenger picked')
            else:
                reward += -10
        # if dropoff action
        elif (action == 5):
            if (self.debug) :
                print()
                print(self.actions_names[action])
            if(not self.check_passenger_in_taxi()):
                reward += -10
            elif( np.array_equal(self.positions [self.positions_names[ self.current_passenger_destination] ] , self.current_taxi_position ) ):
                reward += 20
                finish_episode = True
                if (self.debug) :
                    print('passenger delivered')
            else:
                reward += -10
        if (self.debug) :
            self.plot_map()

        return reward, finish_episode

    def plot_map(self):
        print()
        auxiliar = [list(self.map[i]) for i in range(len(self.map)) ]
        
        pass_dest = self.positions[self.positions_names[self.current_passenger_destination]]
        auxiliar[ pass_dest[0] + 1][1 + 2*pass_dest[1]] = "D"
        if (self.current_passenger_position < len(self.positions_names) -1 ):
            pass_loc = self.positions[self.positions_names[self.current_passenger_position]]
            auxiliar[ pass_loc[0] + 1][1 + 2*pass_loc[1]] = "P"
        auxiliar[ self.current_taxi_position[0] + 1][1 + 2*self.current_taxi_position[1]] = "T"        

        auxiliar2 = [''.join(auxiliar[i])for i in range(len(auxiliar)) ]
        for line in auxiliar2:
            print(line)

        return


class agent():
    '''
    agent that interacts with the environment
    and performs obtain actions based on a 
    softmax policy or greedy. 
    The environment is tabular, so
    q_table is updated in order to 
    obtain the optimal control
    '''
    def __init__(self, grid_size, passenger_loc, passenger_dest, actions, disc_factor, alpha, temperature, method = 'sarsa'):

        self.q_table = np.zeros( shape = [grid_size[0],grid_size[1], passenger_loc, passenger_dest, actions ] )
        self.current_taxi_loc = (0,0)
        self.passenger_loc = 0
        self.passenger_dest = 0
        self.disc_factor = disc_factor
        self.learning_rate = alpha
        self.temperature = temperature

        possible_methods = ['sarsa', 'q_learning', 'expected_sarsa']
        if (method in possible_methods):
            self.method = method
        else:
            print('Wrong method choice')

        return		

    def update_state(self, taxi_loc, passenger_loc, passenger_dest):
        '''
        Update state S
        '''
        self.current_taxi_loc = taxi_loc
        self.passenger_loc = passenger_loc
        self.passenger_dest = passenger_dest
        return

    def get_next_action(self):
        
        '''
        get next action (softmax)
        '''

        action_values = self.q_table[ self.current_taxi_loc[0] ][ self.current_taxi_loc[1] ][ self.passenger_loc] [ self.passenger_dest ]
        softmax = np.exp( action_values / self.temperature )/np.sum( np.exp( action_values / self.temperature ) )
        #next_action = np.argmax(softmax)
        next_action = np.where(np.random.multinomial(1,softmax))[0][0]
        return next_action, action_values

    def get_greedy_action(self):        
        '''
        get next action (greedy)
        '''
        action_values = self.q_table[ self.current_taxi_loc[0] ][ self.current_taxi_loc[1] ][ self.passenger_loc] [ self.passenger_dest ]
        next_action = np.nanargmax(action_values)
        action_value = action_values[next_action]
        
        list_actions = np.where(action_values == action_value)[0]
        if(list_actions.size == 0):
            print("HERE")
        next_action = np.random.choice(list_actions)
        #softmax = np.exp(action_values)/np.sum( np.exp(action_values) )
        #next_action = np.argmax(softmax)
        #next_action = np.where(np.random.multinomial(1,softmax))[0][0]
        return next_action

    def update_state_action_pair(self, current_action, next_taxi_loc, next_passenger_loc, next_passenger_dest, reward):
        '''
        Update q_table
        '''
        current_tax_loc = self.current_taxi_loc
        current_p_loc = self.passenger_loc
        current_state_action_value = self.q_table[current_tax_loc[0]][current_tax_loc[1]][current_p_loc][self.passenger_dest][current_action]

        #Get next state S(t+1)
        self.update_state(next_taxi_loc, next_passenger_loc, next_passenger_dest )
        #Get next action A(t+1)
        new_action, next_action_values = self.get_next_action()

        if(self.method == 'sarsa'):
            self.q_table[current_tax_loc[0]][current_tax_loc[1]][current_p_loc][self.passenger_dest][current_action] = current_state_action_value + self.learning_rate*( reward + 
                                                                                                                                                                        self.disc_factor * self.q_table[ next_taxi_loc[0] ][ next_taxi_loc[1] ][ next_passenger_loc ][ next_passenger_dest ][ new_action ] - 
                                                                                                                                                                        current_state_action_value )
        elif(self.method == 'q_learning'):
            self.q_table[current_tax_loc[0]][current_tax_loc[1]][current_p_loc][self.passenger_dest][current_action] = current_state_action_value + self.learning_rate*( reward + 
                                                                                                                                                                        self.disc_factor * np.max(next_action_values) - 
                                                                                                                                                                        current_state_action_value )            
        elif(self.method == 'expected_sarsa'):
            softmax_prob = np.exp( next_action_values / self.temperature )/np.sum( np.exp( next_action_values / self.temperature ) )
            self.q_table[current_tax_loc[0]][current_tax_loc[1]][current_p_loc][self.passenger_dest][current_action] = current_state_action_value + self.learning_rate*( reward + 
                                                                                                                                                                        self.disc_factor * np.dot(softmax_prob, next_action_values ) - 
                                                                                                                                                                        current_state_action_value )
        

class RL():
    '''
    Class where the agent interacts with the environment
    and receives commands to update the states_action pairs.
    '''

    def __init__ (self, env, agent, max_n_steps, segments, episodes, debug = False, watch_play=True):
        self.env = env
        self.agent = agent

        self.max_n_steps = max_n_steps
        self.segments = segments
        self.episodes = episodes
        
        self.debug = debug
        self.watch_play = watch_play

    def run_episode_training(self):
        #Obtain initial state S
        self.env.initialize_locations()
        self.agent.update_state(self.env.current_taxi_position,
                                self.env.current_passenger_position, 
                                self.env.current_passenger_destination)
        finish_episode = False
        sum_reward = 0                
        iterations = 1

        while (not finish_episode and (iterations <self.max_n_steps)):
            
            #Get a next action A
            action, action_values = self.agent.get_next_action()
            #Observe a reward R
            reward, finish_episode = self.env.perform_action(action)
            #Observe next state S(t+1) and update q_table
            self.agent.update_state_action_pair(action,
                                                self.env.current_taxi_position,
                                                self.env.current_passenger_position,
                                                self.env.current_passenger_destination,
                                                reward)    

            sum_reward += reward*self.agent.disc_factor
                            
            iterations += 1

        return iterations, sum_reward, finish_episode

    def run_episode_greedy(self):
        
        #self.env.debug = True
        self.env.initialize_locations()
        self.agent.update_state(self.env.current_taxi_position,
                            self.env.current_passenger_position, 
                            self.env.current_passenger_destination)
        finish_episode = False
                    
        sum_reward = 0

        iterations =1
        while ( (not finish_episode) and (iterations <self.max_n_steps)):
            action = self.agent.get_greedy_action()
            reward, finish_episode = self.env.perform_action(action)
            self.agent.update_state(self.env.current_taxi_position,
                                    self.env.current_passenger_position,
                                    self.env.current_passenger_destination)
                
            sum_reward += reward*self.agent.disc_factor 
                                                
            iterations += 1
                
        return iterations, sum_reward, finish_episode   

    def train_agent(self):
        '''
        For each segment
        '''
        average_reward_training = 0
        sum_reward_test = 0
        perc_counter = 0

        reward_per_episode = np.zeros(self.segments)
        for segment in range(self.segments):
                    
            '''
            For each episode
            '''
            if( int((segment/self.segments)*100) >= perc_counter):
                print()
                print()
                print('Training episode - segment = {}% complete'.format((segment/self.segments)*100))
                perc_counter+=10

            for episode in range(self.episodes):

                '''
                Training
                '''
                self.env.initialize_locations()
                self.agent.update_state(self.env.current_taxi_position,
                                        self.env.current_passenger_position, 
                                        self.env.current_passenger_destination)
                
                iterations, sum_reward, finish_episode = self.run_episode_training()

                if(self.debug):
                    print('Training episode complete ({}) in {} iterations'.format(finish_episode,iterations))
                    print('Cum reward = {}'.format(sum_reward))
    
            if(segment == self.segments-1):
                average_reward_training += sum_reward / (self.episodes)                           
                            
                self.env.debug = self.watch_play
            iteration_test, sum_reward_test, finish_episode = self.run_episode_greedy()

            if(self.debug):
                print('Testing episode complete {} in {} iterations'.format(finish_episode, iterations))
                print('Cum reward = {}'.format(cum_reward_testing))
        
            
            reward_per_episode[segment] = sum_reward_test
                            
            
            #iteration_test, sum_reward_test, finish_episode = self.run_episode_greedy()


        return average_reward_training, sum_reward_test, reward_per_episode