import numpy as np
import matplotlib.pyplot as plt

def ave_window(running_all,window_length=100):
    average_window = []
    for w1 in range(1,min(len(running_all),window_length)+1):
        average_window.append(np.mean(running_all[:w1]))
    if len(running_all)>window_length:
        for w2 in range(window_length+1,len(running_all)+1):
            average_window.append(np.mean(running_all[w2-window_length:w2]))  
            
    return average_window

def scatter_plot(running_all,y_label,window_length=100,filter_min='',filter_max=''):
    running_all = [x for x in running_all if np.isnan(x) == False]
    if filter_min != '':
        running_all = [x for x in running_all if x > filter_min]
    if filter_max != '':
        running_all = [x for x in running_all if x < filter_max]

    plt.figure(figsize=(25,8))
    plt.scatter([_ for _ in range(1,len(running_all)+1)],running_all)

    plt.plot([*range(1,len(running_all)+1)],running_all,label='Score')
    plt.plot([*range(1,len(running_all)+1)],[0 for _ in range(1,len(running_all)+1)],color='black')
    
    average_window = ave_window(running_all,window_length)

    plt.plot([*range(1,len(running_all)+1)],average_window,color='red',label=f'Average Reward')
    plt.ylabel(y_label)
    plt.xlabel('Episode')

    plt.grid()
    plt.show()
    
def plot_roiTest(test_results):
    plt.figure(figsize=(20,5))
    for k in test_results.keys():
        profits = [v for k,v in test_results[k].items()]
        profits_cumsum = [sum(profits[:i+1]) for i in range(len(profits))]
        plt.plot([*range(1,len(profits_cumsum)+1)],profits_cumsum,label=f'Test {k}')

    plt.xlabel('Expitation Ordered')
    plt.ylabel('Total ROI %')    
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_memoryTime(mem_pressure,time_list):
    plt.figure(figsize=(20,5))
    ax1 = plt.subplot()

    l1, = ax1.plot(time_list,label='Time')
    ax2 = ax1.twinx()
    l2, = ax2.plot(mem_pressure, color='orange',label='Memory')
    ax1.set_ylabel('seconds')
    ax2.set_ylabel('memory %')

    plt.legend()
    plt.show()
    
def tradeCount_plot(tt_total,y_lim=None,inc_bar=True):
    plt.figure(figsize=(20,5))
    if inc_bar == True:
        plt.bar([i for i in range(1,len(tt_total)+1)],tt_total)
    plt.plot([*range(1,len(tt_total)+1)],ave_window(tt_total),color='red',label=f'Average Reward')
    
    plt.ylim(0,y_lim)
    plt.show()