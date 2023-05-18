import pandas as pd
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime

def convert_time2timestamp(t):
    timestamp = time.mktime(t.timetuple())
    return int(timestamp)

def convert_string2datetime(s: str):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def fe(df):
    ## 문자열로 인식되는 Timestamp의 타입을 datetime으로 변경하기. 
    df["Timestamp"] = df["Timestamp"].apply(convert_string2datetime) # string type to datetime type


    ## 기본적인 시간정보 추가
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour                                       # 시간대로 범주 추가 가능


    ## 요일 추가
    df['wday'] = df['Timestamp'].dt.weekday # Monday ~ Sunday => 0 ~ 6         # 주말로 범주 추가 가능
    

    ## 문제를 다시 풀어본 횟수 feature 'retry' 추가
    test_group = df.groupby(['userID','testId']) # 같은 시험지끼리 묶어준다.
    # retry_check = 0
    retry_df = pd.DataFrame()
    for key, group in test_group:
        if len(group[group.assessmentItemID == group.assessmentItemID.iloc[0]]) >= 2:
            retry_df = pd.concat([retry_df,group.groupby('assessmentItemID').cumcount()])
            # retry_check += 1
    retry_df.columns=['retry']
    df = pd.merge(df, retry_df, left_index=True,right_index=True, how="left")
    df['retry'] = df['retry'].fillna(0) # retry의 결측치(한 번만 푼 문제들)을 0으로 바꿔준다.

    
    ## 문제 풀이 시간 추가
    df['time'] = df['time'] = df.groupby(['userID','testId','retry'])['Timestamp'].diff().shift(-1) # 문제 풀이 시간
    df['time'] = df['time'].fillna(df['time'].median())                        # Null값은 중앙값으로 채우기.
    df['time'] = df['time'].apply(lambda x:x.total_seconds())                  # 년,월,일,날짜로 되어있는 값을 시간초로 변환
    df['time'] = df['time'].apply(lambda x:300 if x > 300 else x)              # 최댓값을 300으로 변환.


    ## 문제 풀이 시간 분류 추가
    df['time_class'] = pd.qcut(df['time'],5, labels=[0,1,2,3,4])
    

    ## 전체적인 시간정보를 나타내는 Timestamp는 int형으로 변환.
    df["Timestamp"] = df["Timestamp"].apply(convert_time2timestamp) # datetime to timestamp

    ## 시험지의 평균 정답률, 정답 개수, 표준편차
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum','std'])
    correct_t.columns = ["test_ans_mean", 'test_ans_sum','test_ans_std']
    df = pd.merge(df, correct_t, on=['testId'], how="left")

    ## 시험지의 평균 풀이 시간, 표준편차
    time_t = df.groupby(['testId'])['time'].agg(['mean', 'std'])
    time_t.columns = ["test_time_mean", 'test_time_std']
    df = pd.merge(df, time_t, on=['testId'], how="left")


    ## 시험지 대분류(test_type) 생성
    df['test_type'] = df['testId'].apply(lambda x:x[2])

    ## 시험지 대분류별 정답률, 정답 개수, 표준편차
    correct_type = df.groupby(['test_type'])['answerCode'].agg(['mean', 'sum','std'])
    correct_type.columns = ["t_type_ans_mean", 't_type_ans_sum','t_type_ans_std']
    df = pd.merge(df, correct_type, on=['test_type'], how="left")

    ## 시험지 대분류별 풀이시간, 표준편차
    time_type = df.groupby(['test_type'])['time'].agg(['mean', 'std'])
    time_type.columns = ["t_type_time_mean", 't_type_time_std']
    df = pd.merge(df, time_type, on=['test_type'], how="left")

    ## 시험지 대분류별 난이도
    # 난이도를 나타내는 척도는 임의로 풀이시간 평균 / 평균 정답률로 설정.
    # 풀이시간이 길어지면 난이도가 높아지고, 정답률이 낮아지면 난이도가 높아진다.
    df['test_type_lvl'] = df['t_type_time_mean'] / df['t_type_ans_mean']

    ## tag
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum','std'])
    correct_k.columns = ["tag_mean", 'tag_sum','tag_std']
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")

    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    return df