import pandas as pd
import ujson
import glob
import os
import gzip
from tqdm import tqdm
pd.set_option('display.max_columns',None,'display.max_rows',1000)
from collections import Counter
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import datetime as dt
import pytz
import matplotlib.pyplot as plt

import sys

class OptionEnvironment:
    '''
    There are 3 actions available
    0 : Sell a position (-1)
    1 : Do nothing (0)
    2 : Buy a position (+1)
    
    rewards:
    
    negative points:
    -0.01 points for every day a trade is open
    -0.1 points if the trade is 3 days before expirey
    -0.2 points if the trade is 2 days before expirey
    -0.3 points if the trade is 1 days before expirey
    -1.0 points if the trade is 0 days before expirey
    
    ?-1.0 for trades without a hedge
    
    +0.5 for trades with low deposits per episode
    -0.5 for trades with high deposits per episode
    
    - reward if open interest is less then 100 and reject trade
    
    +%% for trades with positive returns, score calculated with ROI
    -%% for trades with positive returns, score calculated with ROI
    
    -10.0 if there are more -1 trades then +1 trades
    
    -100 if trade exceeds expirey or is on expirey

    '''
    def __init__(self,stock='SPY'):
        self.last_file_reached = False
        self.stock = stock.upper()        
        self.files = sorted(glob.glob(f'Data.nosync/{self.stock}/*.parquet'))
        self.filename_iterator = iter(self.files)
        self.open_trades = {'short':{},'long':{}}
        self.file_generator()
        self.open_interest_limit = 50
        self.reward_index = {'oi_penalty':-1,'0_dte':-10,'short_open_trades':-200,'otr':-100}
        self.historical_trades = []
        self.reward_tracker = {'deposit':[],'pL':[],'open_interest':[],'open_interest':[],'open_trades':[],'Zero_dte_opentrade':[],
                               'days_open':[],'days_open_closed':[],'dte_closed':[],'short_open_trades':[],'force_close_reward':[],'total_reward':[]}
        
        
        
    def file_generator(self):
        # open next time frame as save to self
        try:
            filename = next(self.filename_iterator)
    
            self.filename = filename
            self.df = pd.read_parquet(filename)
            self.df =  self.df[['U_change', 'U_percentChange', 'U_close', 'U_quoteTime', 'U_tradeTime',
                               'U_bid', 'U_ask', 'U_last', 'U_mark', 'U_markChange',
                               'U_markPercentChange', 'U_bidSize', 'U_askSize', 'U_highPrice',
                               'U_lowPrice', 'U_openPrice', 'U_totalVolume', 'U_fiftyTwoWeekHigh',
                               'U_fiftyTwoWeekLow', 'U_price', 'putCall', 'bid', 'ask', 'last', 'mark',
                               'bidSize', 'askSize', 'lastSize', 'highPrice', 'lowPrice', 'openPrice',
                               'closePrice', 'totalVolume', 'tradeTimeInLong', 'netChange',
                               'volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'openInterest',
                               'timeValue', 'theoreticalOptionValue', 'strikePrice', 'expirationDate',
                               'daysToExpiration', 'lastTradingDay', 'percentChange', 'markChange',
                               'markPercentChange', 'inTheMoney', 'expirationType_Q',
                               'expirationType_R', 'quoteTime']]

            try: self.df.drop(columns=['pennyPilot'],inplace=True) ## remove eventually
            except: pass
            try: self.df.drop(columns=['intrinsicValue'],inplace=True) ## remove eventually
            except: pass    


            # append open trades to df
            self.df['openTrades'] = self.df.apply(lambda x: self.insert_open_trades(x.name),axis=1)

            # set df to class state
            self.state = torch.from_numpy(self.df.values).float().to(device)
            self.state_idx = self.df.index

        except:
            print('Last file reached')
            self.last_file_reached = True
            del self.state , self.state_idx, self.df
            
            
    def step(self,actions):
        # select trades that are opened and update the open trades record and rewards
        self.reward_calculator(actions)
        self.file_generator()
        return 
    
    def insert_open_trades(self,idx):
        if idx in self.open_trades['short']:
            return self.open_trades['short'][idx]['positions']
        elif idx in self.open_trades['long']:
            return self.open_trades['long'][idx]['positions']
        else:
            return 0
        
    ### inster the amount of open trades as a feature for all rows, amount of positive and the amount of negative positions

                    
    def reward_calculator(self,actions):                    
        rewards = np.array([])
        self.rewards = rewards
        pL_list = np.array([])
        deposit_list = np.array([])
        dones = np.array([])
        
        self.dor_closingtrade = np.array([])
        self.dte_closingtrade =np.array([])
        Zero_dte_opentrades = np.array([])
        dor_list = np.array([])
        oi_rewards = np.array([])
        force_close_reward_list = np.array([])
        
        
        
        for action,idx,row in zip(actions,self.df.index,self.df.to_dict(orient='records')):
            reward = 0
            pL=0
            deposit = 0
            oi_reward = 0
            dor = 0
            Zero_dte_opentrade=0
            if row['openInterest'] > self.open_interest_limit:
                
                # check if there's an oposing trade open
                if action == 0: # action 0 is a short trade

                    if idx in self.open_trades['long']:
                        reward, pL = self.trade_closer(idx,row,'long',reward)

                    # check if any trades are open and append
                    else:
                        
                        self.dor_closingtrade = np.append(self.dor_closingtrade,0)
                        self.dte_closingtrade = np.append(self.dte_closingtrade,0)

                        deposit += row['bid']
                        date = dt.datetime.fromtimestamp(row['quoteTime'],tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')  
                        if idx in self.open_trades['short']:
                            self.open_trades['short'][idx]['positions'] -= 1
                            self.open_trades['short'][idx]['date_open'].append(date)
                            self.open_trades['short'][idx]['price'].append(row['bid'])

                        # otherwise create new trade index            
                        else:
                            self.open_trades['short'].update({idx:{'positions':-1,'date_open':[date],'price':[row['bid']]}})

                ######## next action
                elif action == 2: # action 2 is a long trade
                    if idx in self.open_trades['short']:
                        reward, pL = self.trade_closer(idx,row,'short',reward)

                    else:
                        self.dor_closingtrade = np.append(self.dor_closingtrade,0)
                        self.dte_closingtrade = np.append(self.dte_closingtrade,0)
                        deposit -= row['bid']
                        date = dt.datetime.fromtimestamp(row['quoteTime'],tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')        
                        # check if any trades are open and append
                        if idx in self.open_trades['long']:
                            self.open_trades['long'][idx]['positions'] += 1
                            self.open_trades['long'][idx]['date_open'].append(date)
                            self.open_trades['long'][idx]['price'].append(row['bid'])
                        # otherwise create new trade index
                        else:
                            self.open_trades['long'].update({idx:{'positions':1,'date_open':[date],'price':[row['bid']]}})      

                # generate reward for trades still open for too long
                elif action == 1: # action 1 is a do nothing
                    self.dor_closingtrade = np.append(self.dor_closingtrade,0)
                    self.dte_closingtrade = np.append(self.dte_closingtrade,0)                    
                    
                    date = dt.datetime.fromtimestamp(row['quoteTime'],tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')  

                    if idx in self.open_trades['short']:
                        open_date = self.open_trades['short'][idx]['date_open'][0]
                        dor = self.days_open_reward(self.ddiff(open_date,date))
                        reward += dor
                        if row['daysToExpiration'] == 0:
                            Zero_dte_opentrade = self.reward_index['0_dte']
                            reward += self.reward_index['0_dte']

                    elif idx in self.open_trades['long']:
                        open_date = self.open_trades['long'][idx]['date_open'][0]
                        dor = self.days_open_reward(self.ddiff(open_date,date))
                        reward+=dor
                        if row['daysToExpiration'] == 0:
                            Zero_dte_opentrade = self.reward_index['0_dte']
                            reward += self.reward_index['0_dte']
                
                
            # if open interest is too low, penalise:
            else:
                self.dor_closingtrade = np.append(self.dor_closingtrade,0)
                self.dte_closingtrade = np.append(self.dte_closingtrade,0)                
                if action != 1:
                    oi_reward = self.reward_index['oi_penalty']                

            
            ## loop through open trades, chck if they were not closed and force close
            force_close_reward = 0
            if (row['daysToExpiration'] == 0):
                if (idx in self.open_trades['long'].keys()) or (idx in self.open_trades['short'].keys()):
                    force_close_reward,force_pL = self.force_close(idx,row)
                    pL += force_pL
                    reward+=force_close_reward
             
            Zero_dte_opentrades = np.append(Zero_dte_opentrades,Zero_dte_opentrade)
            dor_list = np.append(dor_list,dor)         
            oi_rewards = np.append(oi_rewards,oi_reward)            
            force_close_reward_list = np.append(force_close_reward_list,force_close_reward)
            rewards = np.append(rewards,reward)
            pL_list = np.append(pL_list,pL)
            deposit_list = np.append(deposit_list,deposit)        # add pL as reward
            
        rewards = rewards + pL_list
        # negative reward if there are less long positions then short positions, negative reward is -1 for every open contract that short more then long
        # next step maybe seperate trades
        short_open_trades = np.array([0 for x in range(len(rewards))])
        if self.df.openTrades.to_numpy().sum() < 0:    ### speed up this function adds 1.5 seconds -->> reduced to 0.5 seconds with .to_numpy()
            short_open_trades = np.array([self.reward_index['short_open_trades'] for x in range(len(rewards))])
            rewards += short_open_trades

        # negative reward for too many open trades -->> need to add sum of open trades as feauture
        otr = self.open_trades_reward(self.df.openTrades.abs().to_numpy().sum())
        otr = np.array([otr for _ in range(len(rewards))]  )
        rewards = rewards+otr

        # negative points for too large deposits
        sum_deposits = np.sum(np.asarray(deposit_list)) 

        # only include negative reward fro high deposits
        if sum_deposits < 0:
            depsit_reward = np.array([sum_deposits/50 for x in range(len(rewards))])
            rewards += depsit_reward
        else:
            depsit_reward = [0 for x in range(len(rewards))]

        self.rewards = rewards
        self.dones = [True if x == 0 else False for x in self.df['daysToExpiration']]
        
        # reward tracking pL_list
        self.pl = pL_list
        self.reward_tracker['deposit'] += [np.mean(depsit_reward)]
        self.reward_tracker['pL'] += [np.mean(pL_list)]
        self.reward_tracker['open_interest'] += [np.mean(oi_rewards)]
        self.reward_tracker['open_trades'] += [np.mean(otr)]
        self.reward_tracker['days_open'] += [np.mean(dor_list)]
        self.reward_tracker['days_open_closed'] += [np.mean(self.dor_closingtrade)]
        self.reward_tracker['dte_closed'] += [np.mean(self.dte_closingtrade)]
        self.reward_tracker['short_open_trades'] += [np.mean(short_open_trades)]
        self.reward_tracker['Zero_dte_opentrade'] += [np.mean(Zero_dte_opentrades)]
        self.reward_tracker['force_close_reward'] += [np.mean(force_close_reward_list)]

        np.subtract(rewards,short_open_trades)
        np.subtract(rewards,otr)
        np.subtract(rewards,depsit_reward)

        np.add(rewards, np.mean(short_open_trades)).tolist()
        np.add(rewards, np.mean(otr)).tolist()
        np.add(rewards, np.mean(depsit_reward)).tolist()

        self.reward_tracker['total_reward'] += [np.mean(rewards)]
        return

    def ddiff(self,d1,d2):
        d0 = dt.datetime.strptime(d1, "%Y-%m-%d") - dt.datetime.strptime(d2, "%Y-%m-%d")
        return abs(d0.days)

    def dte_reward(self,dte):
        if dte == 3: return -0.1
        if dte == 2: return -0.2
        if dte == 1: return -0.5
        if dte == 0: return -1.0
        else: return 0 

    def days_open_reward(self,days_open, days_no_penalty=6, f=0.03):
        if days_open > days_no_penalty:
            return -sum([0.05**(1-(i*f)) for i in range(days_open-days_no_penalty)])
        else:
            return 0
                            
    def open_trades_reward(self,open_trades, trades_no_penalty=10, f=0.00000005):
        r_=0
        if open_trades > trades_no_penalty:
            for i in range(open_trades-trades_no_penalty):
                r_-=0.004**(1-(i*f))
                if r_ < self.reward_index['otr']:
                    break
        return max(r_,self.reward_index['otr'])
    
    
    def trade_closer(self,idx,row,long_short,reward):
        # closing date
        close_date = dt.datetime.fromtimestamp(row['quoteTime'],tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')  

        open_date = self.open_trades[long_short][idx]['date_open'].pop(0)
        open_price = self.open_trades[long_short][idx]['price'].pop(0)
        close_price = row['ask']
        pL = open_price - close_price
        days_open = self.ddiff(open_date,close_date)
        dte = row['daysToExpiration']
        
        dor_ct = self.days_open_reward(days_open)
        dte_ct = self.dte_reward(dte)

        self.dor_closingtrade = np.append(self.dor_closingtrade,dor_ct)
        self.dte_closingtrade = np.append(self.dte_closingtrade,dte_ct)

        reward += dor_ct
        reward += dte_ct

        if abs(self.open_trades[long_short][idx]['positions']) == 1:
            self.open_trades[long_short].pop(idx)
        else:
            if long_short == 'long':
                self.open_trades[long_short][idx]['positions'] -= 1

            elif long_short == 'short':
                self.open_trades[long_short][idx]['positions'] += 1

        ## add to histprocal trades
        
        
        self.historical_trades.append({'index':idx,'open_date':open_date,'close_date':close_date,'days_open':days_open,'dte before exiting':dte,
                                      'open_price':open_price,'close_price':close_price,'pL':pL,'reward':reward,'force_close':False})

        return reward, pL
    
    
    def force_close(self,idx,row):
        
        for ls in self.open_trades:
            if idx in self.open_trades[ls]:
                for open_date,open_price in zip(self.open_trades[ls][idx]['date_open'],self.open_trades[ls][idx]['date_open']): 
                    r_ = 0
                    if ls == 'long': 
                        close_price = self.df.loc[idx]['bid']
                        pL= close_price - np.sum(self.open_trades[ls][idx]['price'])
                    elif ls == 'short': 
                        close_price = self.df.loc[idx]['ask']
                        pL = (np.sum(self.open_trades[ls][idx]['price'])) - close_price

                    close_date = dt.datetime.fromtimestamp(row['quoteTime'],tz=pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')  
                    days_open = self.ddiff(open_date,close_date)
                    r_-=50
                    if row['openInterest'] < self.open_interest_limit:
                        r_-=25
                    dte = row['daysToExpiration']
                    self.historical_trades.append({'index':idx,'open_date':open_date,'close_date':close_date,'days_open':days_open,'dte before exiting':dte,
                                                    'open_price':open_price,'close_price':close_price,'pL':pL,'reward':r_,'force_close':True})
                    
                # pop all idx
                self.open_trades[ls].pop(idx)
                if idx in self.open_trades[ls]:
                    print('not popped off from force close')
        return r_,pL
