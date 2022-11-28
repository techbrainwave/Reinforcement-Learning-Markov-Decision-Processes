import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
from os import path
import time as tm

from hiive.mdptoolbox import mdp, example, openai
from hiive.mdptoolbox.openai import OpenAI_MDPToolbox as GymEnv

# import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

import gym
import sys

#    MDP: policy : tuple
#         The optimal policy function. Each element is an integer corresponding
#         to an action which maximises the value function in that state.



def p1_vi(st_size=3, pr=False,P=None,R=None):
    var='p1_vi'

    if pr == False:
        P, R = example.forest(st_size)  # default S=3, r1=4, r2=2, p=0.1, is_sparse=False
    vi = mdp.ValueIteration(P, R, 0.96, 0.01, max_iter=1000) # (transitions, reward, discount, epsilon=0.01, max_iter=1000, initial_value=0, skip_check=False)
    # vi = mdp.ValueIteration(P, R, 0.96, 0.0001, max_iter=1000) # (transitions, reward, discount, epsilon=0.01, max_iter=1000, initial_value=0, skip_check=False)
    # vi = mdp.ValueIteration(P, R, 0.96, 0.000000000001, max_iter=1000) # (transitions, reward, discount, epsilon=0.01, max_iter=1000, initial_value=0, skip_check=False)

    vi.verbose=True
    vi.run()
    stats = vi.run_stats
    # print(len(stats))

    print("\nP1 VI:")
    print('Max iter:',vi.max_iter)

    #1 Convergence plot : reward vs iteration
    stats_df= pd.DataFrame([s for s in stats])
    stats_df.plot(x=7, y=2) # Positions of cols
    # plt.ylabel('Reward')
    plt.suptitle("Convergence Plot")
    plt.grid(True)
    plt.savefig(var+str(st_size)+'_convergence.png')     # save plot
    # plt.show()
    plt.close()

    # 2 Policy plot :
    # fl_img = np.reshape(vi.policy, (st_size, 1))
    fl_img = np.reshape(vi.policy, (20, 20))
    fig, ax = plt.subplots()
    ax.imshow(fl_img, interpolation="nearest")
    plt.suptitle("Optimum Policy")
    # plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var + str(st_size) + '_policy.png')  # save plot
    # plt.show()

    print('Size:', st_size, '\nPolicy:',  vi.policy, '\nEpsilon:', vi.epsilon)
    print('Time', vi.time, '\n',vi.error_mean, vi.gamma, vi.v_mean)


def p1_pi(st_size=3,pr=False,P=None,R=None):
    var='p1_pi'

    if pr == False:
        P, R = example.forest(st_size)  # default S=3, r1=4, r2=2, p=0.1, is_sparse=False
    # pi = mdp.PolicyIteration(P, R, 0.9999999999, max_iter=1000) # (transitions, reward, discount, policy0=None, max_iter=1000, eval_type=0, skip_check=False)
    # pi = mdp.PolicyIteration(P, R, 0.999, max_iter=1000) # (transitions, reward, discount, policy0=None, max_iter=1000, eval_type=0, skip_check=False)
    pi = mdp.PolicyIteration(P, R, 0.9, max_iter=1000) # (transitions, reward, discount, policy0=None, max_iter=1000, eval_type=0, skip_check=False)

    pi.verbose=True
    pi.run()
    stats = pi.run_stats
    # print(len(stats))

    print("\nP1 PI:")
    print('Max iter:',pi.max_iter)


    #1 Convergence plot : reward vs iteration
    stats_df= pd.DataFrame([s for s in stats])
    stats_df.plot(x=8, y=2) # Positions of cols
    plt.ylabel('Reward')
    plt.suptitle("Convergence Plot")
    plt.grid(True)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var+str(st_size)+'_convergence.png')     # save plot
    # plt.show()
    plt.close()

    #2 Convergence plot : error vs iteration
    stats_df.plot(x=8, y=3) # Positions of cols
    plt.ylabel('Error')
    plt.suptitle("Convergence Plot")
    plt.grid(True)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var+str(st_size)+'_convergence_err.png')     # save plot
    # plt.show()
    plt.close()

    # 3 Policy plot :
    # fl_img = np.reshape(pi.policy, (st_size, 1))
    fl_img = np.reshape(pi.policy, (20, 20))
    fig, ax = plt.subplots()
    ax.imshow(fl_img, interpolation="nearest")
    plt.suptitle("Optimum Policy")
    # plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var + str(st_size) + '_policy.png')  # save plot
    # plt.show()

    print('Size:', st_size, '\nPolicy:',  pi.policy)#, '\nEpsilon:', pi.epsilon)
    print('Time', pi.time, '\n',pi.error_mean, pi.gamma, pi.v_mean)



def p1_rl(st_size=3,pr=False,P=None,R=None, alpha=0.1,alphad=0.99,eps=1.0,epsd=0.99):
    var = 'p1_rl'

    if pr == False:
        np.random.seed(0)
        P, R = example.forest(st_size)
    # ql = mdp.QLearning(P, R, 0.96, n_iter=300000, run_stat_frequency=1) # (transitions, reward, discount, n_iter=10000, skip_check=False)
    # ql = mdp.QLearning(P, R, 0.96, n_iter=300000, run_stat_frequency=1,alpha=alpha, alpha_decay=alphad,epsilon=eps,epsilon_decay=epsd ) # (transitions, reward, discount, n_iter=10000, skip_check=False)
    ql = mdp.QLearning(P, R, 0.6, n_iter=300000, run_stat_frequency=1,alpha=alpha, alpha_decay=alphad,epsilon=eps,epsilon_decay=epsd ) # (transitions, reward, discount, n_iter=10000, skip_check=False)
    #         (transitions, reward, gamma, alpha = 0.1, alpha_decay = 0.99, alpha_min = 0.001, epsilon = 1.0,
    #         epsilon_min = 0.1, epsilon_decay = 0.99, n_iter = 10000, skip_check = False, iter_callback = None, run_stat_frequency = None)

    ql.verbose=True
    ql.run()
    stats = ql.run_stats


    print("\nP1 RL:")
    print('Max iter:',ql.max_iter)

    # test = stats[0:100]


    # 'State' = {int} 0
    # 'Action' = {int} 0
    # 'Reward' = {float64} 0.0
    # 'Error' = {float64} 0.0
    # 'Time' = {float} 0.0
    # 'Alpha' = {float64} 0.1
    # 'Epsilon' = {float64} 1.0
    # 'Gamma' = {float64} 0.96
    # 'V[0]' = {float64} 0.0
    # 'Max V' = {float64} 0.0
    # 'Mean V' = {float64} 0.0
    # 'Iteration' = {int} 1


    #1 Convergence plot : reward vs iteration
    stats_df= pd.DataFrame([s for s in stats])
    # stats_df= pd.DataFrame([s for s in stats[0:100]])
    stats_df.plot(x=11, y=9) # Positions of cols
    plt.ylabel('Max V')
    # plt.ylabel('Reward')
    plt.suptitle("Convergence Plot")
    plt.grid(True)
    # plt.savefig(var+str(st_size)+'_convergence.png')     # save plot
    plt.savefig(var+str(st_size)+'_'+str(alpha)+'_'+str(alphad)+'_'+str(eps)+'_'+str(epsd)+'_convergence.png')     # save plot
    # plt.show()
    plt.close()

    # 2 Policy plot :
    # fl_img = np.reshape(ql.policy, (st_size, 1))
    fl_img = np.reshape(ql.policy, (20, 20))
    fig, ax = plt.subplots()
    ax.imshow(fl_img, interpolation="nearest")
    plt.suptitle("Optimum Policy")
    # plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var + str(st_size) + '_policy.png')  # save plot
    # plt.show()

    print('Size:', st_size, '\nPolicy:',  ql.policy, '\nEpsilon:', ql.epsilon,'\nAlpha', ql.alpha)
    print('Time', ql.time,'\nGamma:',ql.gamma)#, '\nQ:', ql.Q, '\nV:', ql.V)#,'\nError Mean:',ql.error_mean)
    # print('V Mean:',  ql.v_mean)



def set_episode(s, a, s_new):
    #["SFFF",
    # "FHFH",
    # "FFFH",
    # "HFFG"]
    # 12 = [3][0] = H
    if map_global != None:
        row = s_new//size_global
        col = s_new%size_global
        state_new = map_global[row][col]
        if state_new == 'H' or state_new == 'G':
            return True

    return False



def p1_vi_pi_rl(st_size=0):


    P, R = example.forest(st_size)  # default S=3, r1=4, r2=2, p=0.1, is_sparse=False

    p1_vi(st_size, True, P, R) ## value iteration 3, 20, 400
    p1_pi(st_size, True, P, R)  # Polity iteration
    p1_rl(st_size, True, P, R) # reinforcement learning


    p1_rl(st_size, True, P, R,alpha=0.1,alphad=0.9)
    p1_rl(st_size, True, P, R,alpha=0.5,alphad=0.9)


    p1_rl(st_size, True, P, R,alpha=0.1,alphad=0.9)
    p1_rl(st_size, True, P, R,alpha=0.1,alphad=0.99)
    p1_rl(st_size, True, P, R,alpha=0.2,alphad=0.9)
    p1_rl(st_size, True, P, R,alpha=0.2,alphad=0.99)

    p1_rl(st_size, True, P, R,eps=1.0,epsd=0.99)
    p1_rl(st_size, True, P, R,eps=1.0,epsd=0.9)
    p1_rl(st_size, True, P, R,eps=1.0,epsd=0.5)


def main():


    # print('This message will be displayed on the screen.')

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open('stats_stdout.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        # print('This message will be written to a file.')
        sys.stdout = original_stdout  # Reset the standard output to its original value

    ###########################################3
    # MDP 2 - FOREST - Small
    ###########################################3

        p1_vi(3) # value iteration  # 3,20
        p1_vi(20)
        # p1_vi(50)
        p1_vi(400)

        p1_pi(3) # Polity iteration
        p1_pi(20)
        p1_pi(400)


        p1_rl(3) # reinforcement learning
        p1_rl(20)
        p1_rl(400)


        ## Combined Forest
        p1_vi_pi_rl(3)
        p1_vi_pi_rl(20)
        p1_vi_pi_rl(400)



        print("\n=======")

    # sys.stdout = original_stdout  # Reset the standard output to its original value


if __name__ == "__main__":
    main()