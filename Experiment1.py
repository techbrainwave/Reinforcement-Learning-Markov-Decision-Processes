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


# https://github.com/Farama-Foundation/Gymnasium
# https://github.com/hiive/hiivemdptoolbox/blob/master/hiive/mdptoolbox/openai.py
def p2_vi(st_size=4, map=None):
    var='p2_vi'

    # fl = GymEnv('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    # fl = FrozenLakeEnv(desc=MAPS["8x8"])
    if map is None:
        map = generate_random_map(st_size)
    print(map)
    fl = GymEnv('FrozenLake-v1', desc=map, map_name="nxn", is_slippery=True) # 4x4
    # vi = mdp.ValueIteration(fl.P, fl.R, gamma=.98)                            # If gamma = 1.0 WARNING: check conditions of convergence. With no discount, convergence can not be assumed.
    vi = mdp.ValueIteration(fl.P, fl.R, 0.96, 0.01, max_iter=1000)   # (transitions, reward, discount, epsilon=0.01, max_iter=1000, initial_value=0, skip_check=False)

    vi.verbose=True
    run = vi.run()
    stats = vi.run_stats
    # print(len(stats))

    print(map)
    print("\nP2 VI:")
    print('Max iter:',vi.max_iter)


    #1 Convergence : reward vs iteration
    stats_df= pd.DataFrame([s for s in stats])
    stats_df.plot(x=7, y=2) # Positions of cols
    # plt.ylabel('Reward')
    plt.suptitle("Convergence Plot")
    plt.grid(True)
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.savefig(var+str(st_size)+'_convergence.png')     # save plot
    # plt.show()
    plt.close()


    #2 Policy plot :
    fl_img = np.reshape(vi.policy, (st_size, st_size))  # (16,16)
    fig, ax = plt.subplots()
    ax.imshow(fl_img, interpolation="nearest")
    plt.suptitle("Optimum Policy")
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var+str(st_size)+'_policy.png')     # save plot
    # plt.show()

    print('Size:', st_size, '\nPolicy:',  vi.policy, '\nEpsilon:', vi.epsilon)
    print('Time', vi.time, '\n',vi.error_mean, vi.gamma, vi.v_mean)



def p2_pi(st_size=4, map=None):
    var='p2_pi'

    if map is None:
        map = generate_random_map(st_size)

    fl = GymEnv('FrozenLake-v1', desc=map, map_name="nxn", is_slippery=True)
    # pi = mdp.PolicyIteration(fl.P, fl.R, 0.9999999999, max_iter=1000) # (transitions, reward, discount, policy0=None, max_iter=1000, eval_type=0, skip_check=False)
    pi = mdp.PolicyIteration(fl.P, fl.R, 0.999, max_iter=10) # (transitions, reward, discount, policy0=None, max_iter=1000, eval_type=0, skip_check=False)

    run = pi.run()
    stats = pi.run_stats
    # print(len(stats))


    print("\nP2 PI:")
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

    #3 Policy plot :
    fl_img = np.reshape(pi.policy, (st_size, st_size))  # (16,16)
    fig, ax = plt.subplots()
    ax.imshow(fl_img, interpolation="nearest")
    plt.suptitle("Optimum Policy")
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var+str(st_size)+'_policy.png')     # save plot
    # plt.show()

    print('Size:', st_size, '\nPolicy:',  pi.policy)#, '\nEpsilon:', pi.epsilon)
    print('Time', pi.time, '\n',pi.error_mean,'\n', pi.gamma)#, pi.v_mean)



def p2_rl(st_size=4, map=None,alpha=0.1,alphad=0.99,eps=1.0,epsd=0.99):
    global map_global
    global size_global
    var = 'p2_rl'

    if map is None:
        map = generate_random_map(st_size)
    map_global = map
    size_global = st_size
    fl = GymEnv('FrozenLake-v1', desc=map, map_name="nxn", is_slippery=True) # 4x4
    # ql = mdp.QLearning(fl.P, fl.R, 0.96, n_iter=100000,run_stat_frequency=5)  # (transitions, reward, discount, n_iter=10000, skip_check=False)
    if st_size==16:
        episode=3000000
    else:
        episode=1000000
    # ql = mdp.QLearning(fl.P, fl.R, 0.96, n_iter=episode,run_stat_frequency=5, iter_callback=set_episode)  # (transitions, reward, discount, n_iter=10000, skip_check=False)
    ql = mdp.QLearning(fl.P, fl.R, 0.96, n_iter=episode,run_stat_frequency=5, iter_callback=set_episode,alpha=alpha, alpha_decay=alphad,epsilon=eps,epsilon_decay=epsd )  # (transitions, reward, discount, n_iter=10000, skip_check=False)
    # ql = mdp.QLearning(fl.P, fl.R, 0.96, n_iter=100000000,run_stat_frequency=50)  # (transitions, reward, discount, n_iter=10000, skip_check=False)
    #         (transitions, reward, gamma, alpha = 0.1, alpha_decay = 0.99, alpha_min = 0.001, epsilon = 1.0,
    #         epsilon_min = 0.1, epsilon_decay = 0.99, n_iter = 10000, skip_check = False, iter_callback = None, run_stat_frequency = None)

    ql.verbose = True
    ql.run()
    stats = ql.run_stats

    print("\nP2 RL:")
    print('Max iter EP:', ql.max_iter)


    # 1 Convergence plot : reward vs iteration
    stats_df = pd.DataFrame([s for s in stats])
    # stats_df= pd.DataFrame([s for s in stats[0:100]])
    stats_df.plot(x=11, y=9)  # Positions of cols
    plt.ylabel('Max V')
    plt.xlabel('Episode')
    # plt.ylabel('Reward')
    plt.suptitle("Convergence Plot")
    plt.grid(True)
    # plt.savefig(var + str(st_size) + '_convergenceEP.png')  # save plot
    plt.savefig(var + str(st_size)+'_'+str(alpha)+'_'+str(alphad)+'_'+str(eps)+'_'+str(epsd)+'_convergenceEP.png')  # save plot
    # plt.show()
    plt.close()

    # 2 Policy plot :
    fl_img = np.reshape(ql.policy, (st_size, st_size))  # (16,16)
    fig, ax = plt.subplots()
    ax.imshow(fl_img, interpolation="nearest")
    plt.suptitle("Optimum Policy")
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.savefig(var+str(st_size)+'_policy.png')     # save plot
    # plt.show()

    print('Size:', st_size, '\nPolicy:', ql.policy, '\nEpsilon:', ql.epsilon, '\nAlpha', ql.alpha)
    print('Time', ql.time, '\nGamma:', ql.gamma)  # , '\nQ:', ql.Q, '\nV:', ql.V)#,'\nError Mean:',ql.error_mean)
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


def p2_vi_pi_rl(st_size=0):

    map = generate_random_map(st_size) # Pass the same map

    p2_vi(st_size, map) ## value iteration 16/4
    p2_pi(st_size, map)  # Polity iteration - 4x4, 8x8, 16x16
    p2_rl(st_size, map) # reinforcement learning - 4x4, 8x8, 16x16


    p2_rl(st_size, map,alpha=0.1,alphad=0.9)
    p2_rl(st_size, map,alpha=0.1,alphad=0.99)
    p2_rl(st_size, map,alpha=0.2,alphad=0.9)
    p2_rl(st_size, map,alpha=0.2,alphad=0.99)
    #
    p2_rl(st_size, map,eps=1.0,epsd=0.99)
    p2_rl(st_size, map,eps=1.0,epsd=0.9)
    p2_rl(st_size, map,eps=1.0,epsd=0.5)


def main():


    # print('This message will be displayed on the screen.')

    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open('stats_stdout.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        # print('This message will be written to a file.')
        sys.stdout = original_stdout  # Reset the standard output to its original value


    # ###########################################3
    # # MDP 1 - FROZEN LAKE - Large
    # ###########################################3

        p2_vi(4) ## value iteration 16/4
        p2_vi(8)
        p2_vi(16)

        p2_pi(4)  # Polity iteration - 4x4, 8x8, 16x16
        p2_pi(8)
        p2_pi(16)


        p2_rl(4) # reinforcement learning - 4x4, 8x8, 16x16
        p2_rl(8)
        p2_rl(16)


        # Combined Maps
        p2_vi_pi_rl(4)
        p2_vi_pi_rl(8)
        p2_vi_pi_rl(16)



    # sys.stdout = original_stdout  # Reset the standard output to its original value


if __name__ == "__main__":
    main()