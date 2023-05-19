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


    ## 문제 풀이 시간 그룹 추가
    time_ranges = [-0.001,5,18,27,37,80,300]
    df['time_class'] = pd.cut(df['time'],time_ranges,labels=[0,1,2,3,4,5])
    
    ## 문제 풀이 시간 그룹별 통계량 추가
    time_class_stat = df[df['userID'].shift(1) == df['userID']].groupby(['time_class'])['answerCode'].agg(['mean','sum','std'])
    time_class_stat.columns = ['time_class_mean', 'time_class_sum', 'time_class_std']
    df = pd.merge(df,time_class_stat,on=['time_class'],how='left')


    ## 전체적인 시간정보를 나타내는 Timestamp는 int형으로 변환.
    df["Timestamp"] = df["Timestamp"].apply(convert_time2timestamp) # datetime to timestamp


    ## 시험지의 평균 정답률, 정답 개수, 표준편차
    correct_t = df[df['userID'].shift(1) == df['userID']].groupby(['testId'])['answerCode'].agg(['mean', 'sum','std'])
    correct_t.columns = ["test_ans_mean", 'test_ans_sum','test_ans_std']
    df = pd.merge(df, correct_t, on=['testId'], how="left")

    ## 시험지의 평균 풀이 시간, 표준편차
    time_t = df.groupby(['testId'])['time'].agg(['mean', 'std'])
    time_t.columns = ["test_time_mean", 'test_time_std']
    df = pd.merge(df, time_t, on=['testId'], how="left")


    ## 시험지 대분류(test_type) 생성
    df['test_type'] = df['testId'].apply(lambda x:x[2])

    ## 시험지 대분류별 정답률, 정답 개수, 표준편차
    correct_type = df[df['userID'].shift(1) == df['userID']].groupby(['test_type'])['answerCode'].agg(['mean', 'sum','std'])
    correct_type.columns = ["t_type_ans_mean", 't_type_ans_sum','t_type_ans_std']
    df = pd.merge(df, correct_type, on=['test_type'], how="left")

    ## 시험지 대분류별 풀이시간, 표준편차
    time_type = df.groupby(['test_type'])['time'].agg(['mean', 'std'])
    time_type.columns = ["t_type_time_mean", 't_type_time_std']
    df = pd.merge(df, time_type, on=['test_type'], how="left")

    ## 시험지 난이도
    df['test_lvl'] = df['test_time_mean'] / df['test_ans_mean']
    # 범주화
    cat_num = 10
    df['test_lvl_cat'] = pd.qcut(df['test_lvl'],cat_num,labels=[i for i in range(cat_num)])

    ## 시험지 대분류별 난이도
    # 난이도를 나타내는 척도는 임의로 풀이시간 평균 / 평균 정답률로 설정.
    # 풀이시간이 길어지면 난이도가 높아지고, 정답률이 낮아지면 난이도가 높아진다.
    df['test_type_lvl'] = df['t_type_time_mean'] / df['t_type_ans_mean']

    # 범주화
    type_cat_num = 3
    df['test_type_lvl_cat'] = pd.qcut(df['test_type_lvl'],type_cat_num,labels=[i for i in range(type_cat_num)])

    ## 시험지 노출 횟수
    df['test_total_answer'] = df.groupby('testId')['answerCode'].cumcount()


    ## tag의 평균 정답률, 정답 총합, 표준편차
    correct_k = df[df['userID'].shift(1) == df['userID']].groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum','std'])
    correct_k.columns = ["tag_ans_mean", 'tag_ans_sum','tag_ans_std']
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    ## 태그의 평균 풀이 시간, 표준편차
    time_k = df.groupby(['KnowledgeTag'])['time'].agg(['mean', 'std'])
    time_k.columns = ["tag_time_mean", 'tag_time_std']
    df = pd.merge(df, time_k, on=['KnowledgeTag'], how="left")

    ## 태그 난이도
    df['tag_lvl'] = df['tag_time_mean'] / df['tag_ans_mean']

    # 범주화
    tag_cat_num = 10
    df['tag_lvl_cat'] = pd.qcut(df['tag_lvl'],tag_cat_num,labels=[i for i in range(tag_cat_num)])

    ## 태그 노출 횟수
    df['tag_total_answer'] = df.groupby('KnowledgeTag')['answerCode'].cumcount()


    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_ans_1'] = df[df['userID'].shift(1) == df['userID']].groupby('userID')['answerCode'].transform(lambda x:x.cumsum().shift(1))
    df['user_total_ans'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_ans_1']/df['user_total_ans']

    df['user_test_ans_count'] = df.groupby(['userID','testId'])['answerCode'].cumcount()
    df['user_test_ans_1'] = df[df['userID'].shift(1) == df['userID']].groupby(['userID','testId'])['answerCode'].transform(lambda x:x.cumsum().shift(1))
    df['user_test_acc'] = df['user_test_ans_1'] / df['user_test_ans_count']
    df['user_test_lvl_mean'] = df.groupby(['userID'])['test_lvl'].cumsum() / (df.user_total_ans + 1)

    df['user_tag_ans_count'] = df.groupby(['userID','KnowledgeTag'])['answerCode'].cumcount()
    df['user_tag_ans_1'] = df[df['userID'].shift(1) == df['userID']].groupby(['userID','KnowledgeTag'])['answerCode'].transform(lambda x:x.cumsum().shift(1))
    df['user_tag_acc'] = df['user_tag_ans_1'] / df['user_tag_ans_count']
    df['user_tag_lvl_mean'] = df.groupby(['userID'])['tag_lvl'].cumsum() / (df.user_total_ans + 1)

    df['user_lvl'] = df.user_tag_lvl_mean + df.user_test_lvl_mean


    df['item_ans_1'] = df[df['userID'].shift(1) == df['userID']].groupby('assessmentItemID')['answerCode'].cumsum()
    df['item_total_ans'] = df.groupby('assessmentItemID')['answerCode'].cumcount() + 1
    df['item_acc'] = df['item_ans_1']/df['item_total_ans']

    
    ## item의 평균 정답률, 정답 총합, 표준편차
    correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum','std'])
    correct_a.columns = ["item_ans_mean", 'item_ans_sum','item_ans_std']
    df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
    
    ## item의 평균 풀이 시간, 표준편차
    time_a = df.groupby(['assessmentItemID'])['time'].agg(['mean', 'std'])
    time_a.columns = ["item_time_mean", 'item_time_std']
    df = pd.merge(df, time_a, on=['assessmentItemID'], how="left")

    ## item 난이도
    df['item_lvl'] = df['item_time_mean'] / df['item_ans_mean']
    # 범주화
    item_cat_num = 10
    df['item_lvl_cat'] = pd.qcut(df['item_lvl'],item_cat_num,labels=[i for i in range(item_cat_num)])

    ## item 노출 횟수
    df['item_total_answer'] = df.groupby('assessmentItemID')['answerCode'].cumcount()
    
    df = df.fillna(0)
    
    return df