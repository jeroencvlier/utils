class OTGym_v1(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(OTGym,self).__init__()
        self.newRecord = {'tradePosition':0,'deposit':0,
                          'pL':0,'roi':0,'runningPL':0,'runningRoi':0,
                         'daysOpen':0,'sequencesOpen':0} 

        self.idx_list = []
        self.totalroi = []
        self.totalpl = []
        self.sqz_open = []
        self.tradeCount = 0
        self.current_step = 0
        self.historicalWindow = 5
        
        if os.path.exists('Data.nosync/SPY/stableBaselines/'):
            self.data_directory = 'Data.nosync/SPY/stableBaselines/'
            self.minmax = MinMax2(self.data_directory,reset=False)
            self.env_location = 'mac'
        else:
            !pip install tianshou -q
            from google.colab import drive
            
            drive.mount('/content/drive')
            self.data_directory = '/content/drive/MyDrive/stableBaselines/'
            self.minmax = MinMax3()
            self.env_location = 'colab'
            
        self.min_deposit = self.minmax.minimum[self.minmax.columns.index('deposit_mark')]
        self.max_deposit = self.minmax.maximum[self.minmax.columns.index('deposit_mark')]
        self.min_tradePosition = 0
        self.max_tradePosition = 1
        self.min_pL = -25
        self.max_pL = 25
        self.min_Roi = -1
        self.max_Roi = 10
        self.min_sqz = 10
        self.max_sqz = 445
        self.min_days_open = self.minmax.minimum[self.minmax.columns.index('daysToExpiration_front')]
        self.max_days_open = self.minmax.maximum[self.minmax.columns.index('daysToExpiration_front')]


        self.files = sorted(glob.glob(self.data_directory+'*'))
            
        self.fileUsage = {x:1 for x in self.files}
        self.subFileUsage = {x:{} for x in self.files}
        
        self.width = pd.read_parquet(glob.glob(self.data_directory+'*')[0]).shape[-1] + len(self.newRecord)
        
        self.observation_space = Box(low=-1, high=1, shape=(self.historicalWindow, self.width))
        self.action_space = Discrete(2)   
        
    def _next_observation(self,):
        self.scaler_trade_record()
        ts_t = torch.Tensor([[np.float32(v) for k,v in ts.items()] for ts in self.trade_state_list_scaled])
        self.obs = torch.concat((self.minmax.scale(self.dflocked[self.current_step-self.historicalWindow:self.current_step]),ts_t),dim=1)
        
    def _take_action(self,):
        self.reward = 0
        if self.action == 1:
            if self.trade_record['tradePosition'] == 0:
                self.trade_record['tradePosition'] = 1
                self.tradeCount +=1
                self.db4exp = self.dflocked.iloc[self.current_step]['daysToExpiration_front']
                #self.trade_record['deposit'] = self.dflocked.iloc[self.current_step]['deposit_mark'] 
                self.trade_record['deposit'] = round(random.uniform(self.dflocked.iloc[self.current_step]['bid_back'],self.dflocked.iloc[self.current_step]['ask_back']) - \
                                                     random.uniform(self.dflocked.iloc[self.current_step]['bid_front'],self.dflocked.iloc[self.current_step]['ask_front']),2)
                if self.trade_record['deposit'] < 0.01:
                    self.trade_record['deposit']=0.01
                                    
                if self.dflocked.iloc[self.current_step]['openInterest_front'] < 50:
                    self.reward -= 1000
                    
                if self.dflocked.iloc[self.current_step]['openInterest_back'] < 50:
                    self.reward -=1000
                    
                # if abs(self.dflocked.iloc[self.current_step]['delta_front']) < 0.3: self.reward -=1000
                # if abs(self.dflocked.iloc[self.current_step]['delta_back']) < 0.7: self.reward -=1000                       
                    

            elif self.trade_record['tradePosition'] == 1:
                self.trade_record['sequencesOpen'] += 1
                self.trade_record['daysOpen'] = self.db4exp - self.dflocked.iloc[self.current_step]['daysToExpiration_front']
                
                self.trade_record['pL'] = np.float32(self.dflocked.iloc[self.current_step]['deposit_mark'] - self.trade_record['deposit'])
                # self.trade_record['pL'] = round(random.uniform(self.dflocked.iloc[self.current_step]['bid_back'],self.dflocked.iloc[self.current_step]['ask_back']) - \
                #                                 random.uniform(self.dflocked.iloc[self.current_step]['bid_front'],self.dflocked.iloc[self.current_step]['ask_front']),2) - self.trade_record['deposit']

                self.trade_record['roi'] = np.float32(self.trade_record['pL']/self.trade_record['deposit'])

                self.trade_record['runningPL'] = np.float32(self.trade_record['pL']/self.trade_record['sequencesOpen'])
                self.trade_record['runningRoi'] = np.float32(self.trade_record['roi']/self.trade_record['sequencesOpen'])

                self.reward += self.trade_record['runningPL']*100
                # self.reward += self.trade_record['runningRoi']*100
                
                if self.done[self.current_step] == True:
                    self.reward -= 10000
                    
        elif self.action == 0:
            if self.trade_record['tradePosition'] == 1:
                self.sqz_open.append(self.trade_record['sequencesOpen'])
                self.trade_record['tradePosition'] = 0
                self.trade_record['pL'] = round(random.uniform(self.dflocked.iloc[self.current_step]['bid_back'],self.dflocked.iloc[self.current_step]['ask_back']) - \
                                                random.uniform(self.dflocked.iloc[self.current_step]['bid_front'],self.dflocked.iloc[self.current_step]['ask_front']),2) - self.trade_record['deposit']                
                
                self.trade_record['roi'] = self.trade_record['pL']/self.trade_record['deposit']    
                
                self.roi += self.trade_record['roi']
                self.pl += self.trade_record['pL']                       
                
                self.trade_record['deposit'] = 0
                # self.reward += self.trade_record['roi']*100
                # self.reward += self.trade_record['pL']*100

                self.trade_record['pL'] = 0
                self.trade_record['roi'] = 0

                self.trade_record['daysOpen'] = 0
                self.trade_record['sequencesOpen'] = 0
                self.trade_record['runningPL'] = 0
                self.trade_record['runningRoi'] = 0  
                self.db4exp = 0

            elif self.trade_record['tradePosition'] == 0:
                pass     
            
    def scaler_trade_record(self,):
        self.trade_state_list_scaled =[{key: value for key, value in x.items()} for x in self.trade_state_list] #self.trade_state_list.copy()
        for d in self.trade_state_list_scaled:
            d['tradePosition'] = (((d['tradePosition'] - self.min_tradePosition) / (self.max_tradePosition - self.min_tradePosition))*2)-1   
            d['deposit'] = (((d['deposit'] - self.min_deposit) / (self.max_deposit - self.min_deposit))*2)-1
            d['pL'] = (((d['pL'] - self.min_pL) / (self.max_pL - self.min_pL))*2)-1
            d['roi'] = (((d['roi'] - self.min_Roi ) / (self.max_Roi - self.min_Roi ))*2)-1
            d['runningPL'] = (((d['runningPL'] - self.min_pL) / (self.max_pL - self.min_pL))*2)-1    
            d['runningRoi'] = (((d['runningRoi'] - self.min_Roi ) / (self.max_Roi - self.min_Roi ))*2)-1    
            d['daysOpen'] = (((d['daysOpen'] - self.min_days_open) / (self.max_days_open - self.min_days_open))*2)-1    
            d['sequencesOpen'] = (((d['sequencesOpen'] - self.min_sqz ) / (self.max_sqz  - self.min_sqz ))*2)-1    
            
            

    def step(self, action):    
        self.current_step +=1
        self.action = action
        self._take_action()
        d = {key: value for key, value in self.trade_record.items()}   # does a deep copy
        self.trade_state_list.append(d)
        self._next_observation()
        
        if self.done[self.current_step] == True:
            self.totalroi.append(self.roi)
            self.totalpl.append(self.pl)
            
        # display(pd.DataFrame(self.dflocked.iloc[self.current_step]).T[['WD','HOUR','daysToExpiration_front','bid_front','mark_front','ask_front','bid_back','mark_back','ask_back','deposit_mark','openInterest_front','openInterest_back']])
        # print(self.trade_record)
        
        self.info = {}
        # if self.done[self.current_step] == True:
        #     self.info = {'No. of Trades': self.tradeCount,'Sequences Open': self.sqz_open,
        #                 'Total ROI': self.roi,'Total PL':self.totalpl}
        return self.obs, self.reward, self.done[self.current_step], self.info
        

    def reset(self):
        self.reward = 0
        self.trade_record=self.newRecord    
        self.db4exp = 0
        self.pl = 0
        self.roi = 0
        self.trade_state_list = deque(maxlen=5)
        d = {key: value for key, value in self.newRecord.items()}  # makes a deep copy of the dictionary
        [self.trade_state_list.append(d) for _ in range(5)]
        
        while True:
            if len(self.idx_list) == 0:
                p_ = sum([v for k,v in self.fileUsage.items()])
                self.file = np.random.choice(list(self.fileUsage.keys()),p=[(v/p_) for k,v in self.fileUsage.items()])
                for k,v in self.fileUsage.items():
                    if k != self.file:
                        self.fileUsage[k]+=1
                self.df = pd.read_parquet(self.file)
                self.idx_list = list(self.df.index.unique())

            self.idx = np.random.choice(self.idx_list)
            self.dflocked = self.df.loc[self.idx].sort_values(['daysToExpiration_front','HOUR'],ascending=[False,True])  
            self.idx_list.remove(self.idx)
            if len(self.dflocked)>20:
                break
            
        self.current_step = self.historicalWindow    
        self.scaler_trade_record()
        ts_t = torch.Tensor([[np.float32(v) for k,v in ts.items()] for ts in self.trade_state_list_scaled])
        self.obs = torch.concat((self.minmax.scale(self.dflocked[0:self.historicalWindow]),ts_t),dim=1)
        
        self.done = [False for _ in range(self.dflocked.shape[0]-1)]+[True]
        self.max_steps = len(self.dflocked)
        return self.obs 
        
    def render(self, mode='human', close=False):
        if len(self.totalpl)==0: tplmean = np.mean(self.totalpl)
        else:tplmean = 0
            
        if len(self.totalpl)==0: troimean = np.mean(self.totalroi)   
        else:troimean = 0
            
        print(f'Average PL: {round(tplmean,2)}, Average ROI: {round(troimean,2)}, Total Trades {self.tradeCount}')
