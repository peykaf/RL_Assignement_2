import taxi_environment
import numpy as np
import matplotlib.pyplot as plt

'''
Parameters
'''

saveFigures = True
plot_graph_ = False
debug_training = False
debug_testing = False
plot_map = False

indepedent_runs = 2
segments = 5
episodes = 10
max_n_steps = 200#100


MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
grid_size = (5,5)


temperatures = [ 0.6, 1.0, 1.5]#[ 0.3, 0.6, 1.0, 1.5, 4] #[0.5, 0.55, 0.6, 0.65, 0.7, 0.8] #[0.1 , 0.5 , 0.9]
learning_rate = [0.25, 0.5, 0.8, 1.0] #[0.1,0.8, 0.9,1]#[0.3 , 0.4 , 0.6]
disc_factor = .9

temp_names = [str(i) for i in temperatures]
l_rate_names = [str(i) for i in learning_rate]
method = ['sarsa', 'q_learning', 'expected_sarsa']#['expected_sarsa']#['sarsa']#['q_learning']


def plot_graph(x_value, y_values, legend_names, x_axis_title, y_axis_title, filename_to_save):

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    for i in range(len(y_values)):
        ax.plot(x_value,
                y_values[i],
                label = legend_names[i])
    ax.set_xlabel (x_axis_title)
    ax.set_ylabel (y_axis_title)

    #ax.set_ylim( -150, 20)

    plt.legend(loc = 'upper left')
    if(saveFigures):
        plt.savefig(filename_to_save)
    if (plot_graph_):
        plt.show()    
    plt.clf()
    
    return


def main():
                 
    for meth in method:
        
        agent_return_training = np.zeros( shape = ( len(temperatures) , len(learning_rate) ) )
        agent_return_training2 = np.zeros( shape = ( len(learning_rate) , len(temperatures) ) )

        agent_return_testing = np.zeros( shape = ( len(temperatures) , len(learning_rate) ) )
        agent_return_testing2 = np.zeros( shape = ( len(learning_rate) , len(temperatures) ) )
        
        temp = 0
        for temperature in temperatures:
            l_rate = 0
            for alpha in learning_rate:
                """
                For indepedent runs
                """
                reward_per_episode_av = np.zeros(segments)
                for run in range(indepedent_runs):
                    print('Initializing run {}'.format(run) )
                    print('method = '+ meth)
                    print('temperature = {}'.format(temperature))
                    print('learning_rate = {}'.format(alpha))

                    map = taxi_environment.environment( grid_size, MAP, debug = plot_map )

                    agent = taxi_environment.agent(grid_size, 
                                                      len(map.position_index), 
                                                      len(map.position_index) - 1 , 
                                                      len(map.actions), 
                                                      disc_factor, 
                                                      alpha,
                                                      temperature,
                                                      method = meth )
                    
                    rl = taxi_environment.RL(map, agent, max_n_steps, segments, episodes)

                    rw_train, rw_test, reward_per_episode = rl.train_agent()

                    agent_return_training[temp][l_rate] += rw_train / (indepedent_runs)
                    agent_return_training2[l_rate][temp] += rw_train / (indepedent_runs)
                    agent_return_testing[temp][l_rate] += rw_test / (indepedent_runs)
                    agent_return_testing2[l_rate][temp] += rw_test / (indepedent_runs)

                    reward_per_episode_av += reward_per_episode/indepedent_runs

                    continue


                plot_graph(np.arange(segments)+1, 
                           [reward_per_episode_av], 
                           ['l_rate = '+str(alpha)+'/n temp = '+str(temperature)], 
                           'number of segments',
                           'Av. return ('+meth+')',
                           'learning_curve'+meth+'_temp'+str(temperature)+'_alpha'+str(alpha)+'.png')

                l_rate+=1
            temp+=1
                    


        """
        Plot training curves
        """    
        plot_graph(learning_rate, 
                   agent_return_training,
                   temp_names, 
                   'learning rate',
                   'Av. return during training ('+meth+')',
                   'av_return_l_rate_training_'+meth+'.png')             

        #plot_graph(temperatures, 
        #           agent_return_training2, 
        #           l_rate_names, 
        #           'temperature',
        #           'Av. return during training ('+meth+')',
        #           'av_return_temp_training_'+meth+'.png')
        
                
        """
        Plot testing curves
        """    
        plot_graph(learning_rate, 
                   agent_return_testing, 
                   temp_names, 
                   'learning rate',
                   'Av. return during testing ('+meth+')',
                   'av_return_l_rate_testing_'+meth+'.png')             

        #plot_graph(temperatures, 
        #           agent_return_testing2, 
        #           l_rate_names, 
        #           'temperature',
        #           'Av. return during testing ('+meth+')',
        #           'av_return_temp_testing_'+meth+'.png')

        
        
        temp = 0
        for temperature in temperatures:
            plot_graph(learning_rate, 
                        [agent_return_training[temp], agent_return_testing[temp] ], 
                        ['training','testing'], 
                        'temperature',
                        'Av. return during testing ('+meth+'), temperature = '+str(temperature)+')',
                        'av_return_l_rate_train_test_'+meth+'temp_'+str(temperature)+'.png')
                
            temp += 1

        #l_rate = 0
        #for alpha in learning_rate:
        #        plot_graph(temperatures, 
        #                [agent_return_training2[l_rate], agent_return_testing2[l_rate] ], 
        #                ['training','testing'], 
        #                'temperature',
        #                'Av. return during testing ('+meth+'), l_rate = '+str(alpha)+')',
        #                'av_return_temp_train_test_'+meth+'l_rate_'+str(alpha)+'.png')
        #        l_rate += 1

    return

if __name__ == "__main__":
    main()