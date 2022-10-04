import pandas as pd
import torch
import numpy as np
def train_test_loop(en1,filename,test_net,minmax,trading_fee):
    
    df = pd.read_parquet(filename)
    if len(df)!=0:
        df_roi = []
        states_list = [df.loc[calendar_id] for calendar_id in list(set(df.index))]
        for states in states_list:
            states = states.sort_values(['daysToExpiration_front','WD','HOUR'],ascending=[False,True,True]).copy()
            states_scaled = minmax.scale(states)  # scale the data 

            dones = [False for _ in range(len(states_scaled)-2)] + [True] 
                
            last_action = 0.0
            open_deposit = 0.0
            pl = 0.0
            running_pl = 0.0
            days_open=0.0
            new_trade = True
            sqz_open = 0.0
            state_roi = []

            for en,done in enumerate(dones):

                state = torch.cat((states_scaled[en], torch.FloatTensor([last_action,minmax.value_scale(open_deposit,'deposit_mark'),pl,running_pl,days_open,float(new_trade),sqz_open])), 0).unsqueeze(0)
                action = test_net(state)
                if (torch.argmax(action) == 1) and (done==False):

                    # If it is a new trade, initialise the record keeping
                    if new_trade == True:
                        new_trade = False
                        dte_initial = states['daysToExpiration_front'][en]
                        open_deposit = states['deposit_mark'][en]


                    # If the trade is still open we update the tracking
                    elif new_trade == False:
                        days_open = dte_initial - states['daysToExpiration_front'][en]
                        sqz_open +=1
                        running_pl = states['deposit_mark'][en] - open_deposit


                #check if open trade is closed
                if (torch.argmax(action) == 0) or (done == True):
                    if new_trade == False:
                        new_trade = True
                        reset_pl=True

                        # calculate actual pl
                        # to avoid calculations where there is not enough data we avoid trades that do not have all the data
                        if done == True:
                            if states['daysToExpiration_front'][en]==0:
                                pl = states['deposit_mark'][en] - open_deposit - trading_fee
                        else:
                            pl = states['deposit_mark'][en] - open_deposit - trading_fee
                        
                        # to avoid division by 0, make minimum deposit $10
                        if open_deposit < 0.05:
                            roi = pl/0.05
                        else:
                            roi = pl/open_deposit

                        state_roi.append(roi)

                        running_pl = states['deposit_mark'][en] - open_deposit

                        open_deposit = 0.0
                        days_open = 0.0
                        sqz_open = 0.0
            
            if len(state_roi) != 0:
                state_roi = np.mean(state_roi)
                df_roi.append(state_roi)
        
        if len(df_roi)!=0:
            df_roi = round(np.mean(df_roi),4)
        else:
            df_roi = 0.0
            
        return {en1:df_roi}


