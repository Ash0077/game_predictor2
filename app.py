
import urllib 
import pandas as pd
from datetime import datetime
import numpy as np
from pandas import DataFrame
import plotly.express as px
from flask import Flask,json, request, jsonify
from flask_cors import CORS
import os
import glob
from scipy import stats
app= Flask(__name__)
CORS(app)

@app.route('/predict/<select_league>/<ht>/<at>', methods=['GET'])
def PredictScore(select_league,ht,at,):
    folder='datasets/'+select_league+'/'
    raw_data_1 = pd.read_csv(folder +'season-0809.csv')
    raw_data_2 = pd.read_csv(folder +'season-0910.csv')
    raw_data_3 = pd.read_csv(folder +'season-1011.csv')
    raw_data_4 = pd.read_csv(folder +'season-1112.csv')
    raw_data_5 = pd.read_csv(folder +'season-1213.csv')
    raw_data_6 = pd.read_csv(folder +'season-1314.csv')
    raw_data_7 = pd.read_csv(folder +'season-1415.csv')
    raw_data_8 = pd.read_csv(folder +'season-1516.csv')
    raw_data_9 = pd.read_csv(folder +'season-1617.csv')
    raw_data_10 = pd.read_csv(folder +'season-1718.csv')
    raw_data_11 = pd.read_csv(folder +'season-1819.csv')
    raw_data_12 = pd.read_csv(folder +'season-1920.csv')
    raw_data_13 = pd.read_csv(folder +'season-2021.csv')
    raw_data_14 = pd.read_csv(folder +'season-2122.csv')
    os.chdir('datasets/'+select_league+'/')
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
    df=pd.read_csv("combined_csv.csv")
    df=df.fillna(0)
    dataset=df[["Date","HomeTeam",'AwayTeam','FTHG','FTAG','HC','AC','HY','AY','HR','AR']]
    dataset['total_goals']=dataset['FTHG']+dataset['FTAG']
    dataset.iloc[[dataset.total_goals.argmax()]]
    # print(dataset.total_goals.mean())

    if len(dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)]) > 20:
        
        avg_FTHG = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].FTHG.mean()
        avg_FTAG = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].FTAG.mean()
        ###HOME CORNER
        avg_HC = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].HC.mean()
        ####AWAY CORNER
        avg_AC = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].AC.mean()
        #home yellow
        avg_HY = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].HY.mean()
        #away yellow
        avg_AY = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].AY.mean()
        
        #home red
        avg_HR = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].HR.mean()
        #away red
        avg_AR = dataset[(dataset.HomeTeam ==ht) & (dataset.AwayTeam ==at)].AR.mean()
        
        
        
        
        #RED
        away_red=int(stats.mode(np.random.poisson(avg_AR,100000))[0])
        home_red=int(stats.mode(np.random.poisson(avg_HR,100000))[0])
        
        
        #yellow
        away_yellow=int(stats.mode(np.random.poisson(avg_AY,100000))[0])
        home_yellow=int(stats.mode(np.random.poisson(avg_HY,100000))[0])
        #corner
        away_corner=int(stats.mode(np.random.poisson(avg_AC,100000))[0])
        home_corner=int(stats.mode(np.random.poisson(avg_HC,100000))[0])  
        #goal
        home_goal = int(stats.mode(np.random.poisson(avg_FTHG,100000))[0])                    
        away_goal = int(stats.mode(np.random.poisson(avg_FTAG,100000))[0])
        
    else:
        
        avg_home_goal_conceded = dataset[(dataset.HomeTeam ==ht)].FTAG.mean()
        avg_away_goal_scored   = dataset[(dataset.AwayTeam ==at)].FTAG.mean()
        away_goal = int(stats.mode(np.random.poisson(1/2*(avg_home_goal_conceded+avg_away_goal_scored),100000))[0])
        
        avg_away_goal_conceded = dataset[(dataset.HomeTeam ==at)].FTHG.mean()
        avg_home_goal_scored   = dataset[(dataset.AwayTeam ==ht)].FTHG.mean()
        home_goal = int(stats.mode(np.random.poisson(1/2*(avg_away_goal_conceded+avg_home_goal_scored),100000))[0])
    
        
        #corner
        avg_home_corner_conceded = dataset[(dataset.HomeTeam ==ht)].AC.mean()
        avg_away_corner_scored   = dataset[(dataset.AwayTeam ==at)].AC.mean()
        away_corner = int(stats.mode(np.random.poisson(1/2*(avg_home_corner_conceded+avg_away_corner_scored),100000))[0])
        
        #corner
        avg_away_corner_conceded = dataset[(dataset.HomeTeam ==at)].HC.mean()
        avg_home_corner_scored   = dataset[(dataset.AwayTeam ==ht)].HC.mean()
        home_corner = int(stats.mode(np.random.poisson(1/2*(avg_away_corner_conceded+avg_home_corner_scored),100000))[0])
        
        # away yellow
        
        avg_home_yellow_conceded = dataset[(dataset.HomeTeam ==ht)].AY.mean()
        avg_away_yellow_scored   = dataset[(dataset.AwayTeam ==at)].AY.mean()
        away_yellow = int(stats.mode(np.random.poisson(1/2*(avg_home_yellow_conceded+avg_away_yellow_scored),100000))[0])
        # home yellow
        
        avg_away_yellow_conceded = dataset[(dataset.HomeTeam ==at)].HY.mean()
        avg_home_yellow_scored   = dataset[(dataset.AwayTeam ==ht)].HY.mean()
        home_yellow = int(stats.mode(np.random.poisson(1/2*(avg_away_yellow_conceded+avg_home_yellow_scored),100000))[0])
        
        
        # away RED
        
        avg_home_red_conceded = dataset[(dataset.HomeTeam ==ht)].AR.mean()
        avg_away_red_scored   = dataset[(dataset.AwayTeam ==at)].AR.mean()
        away_red = int(stats.mode(np.random.poisson(1/2*(avg_home_red_conceded+avg_away_red_scored),100000))[0])
        # home yellow
        
        avg_away_red_conceded = dataset[(dataset.HomeTeam ==at)].HR.mean()
        avg_home_red_scored   = dataset[(dataset.AwayTeam ==ht)].HR.mean()
        home_red = int(stats.mode(np.random.poisson(1/2*(avg_away_red_conceded+avg_home_red_scored),100000))[0])
        
        
        
        
        
    booking_points_home= home_yellow*10+  home_red*25
    booking_points_away= away_yellow*10+  away_red*25
    
    avg_total_score = int(stats.mode(
        np.random.poisson((dataset[(dataset.HomeTeam==ht) &    (dataset.AwayTeam==at)].total_goals.mean()),100000))[0])
    
    temp={"HomeTeam_GOAL":home_goal,"AwayTeam_GOAL":away_goal,
                       "HomeTeam_corner":home_corner,"AwayTeam_corner":away_corner,
                       "HomeTeam_yell_card":home_yellow,"AwayTeam_yell_card":away_yellow,
                       "HomeTeam_red_card":home_red,"AwayTeam_red_card":away_red,
                      "HomeTeam_booking_point":booking_points_home,"AwayTeam_booking_point":booking_points_away
                      }
    
    return json.dumps(temp)

if __name__== "__main__":
     app.run()
