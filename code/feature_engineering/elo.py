import pandas as pd
import numpy as np
from math import log10 as log



train_data = pd.read_csv('./data/train_data.csv')
test_data = pd.read_csv('./data/test_data.csv')
df = pd.concat([train_data, test_data], ignore_index=True)

train = df[df.answerCode != -1]
test = df[df.answerCode == -1]

problems = train.assessmentItemID.unique()
students = train.userID.unique()

# rate는 4000으로 시작.
problem_rate = {problem:4000 for problem in problems}
student_rate = {student:4000 for student in students}
df = train[['userID', 'assessmentItemID', 'answerCode']]

K = 1
# ELO Rating function
def win_rate(p_op, p_me): # 문제를 풀었을 때 얻는 rating 점수
    return 1/(10**((p_op-p_me)/100)+1) 

def changed_score(K, p_op, p_me, result):
    '''
    params
        - K : Rating 변경 비율
        - p_op : assessmentID (문제)
        - p_me : UserID (사용자)
        - result : answerCode (문제 정오답 여부)
    variable
        - game_percent : 풀이로부터 변경되는 Rating 값.
        - next_op : 문제의 갱신된 Rating
        - next_me : User의 갱신된 Rating
    '''
    game_pecent = win_rate(p_op, p_me)
    next_op = p_op + K*(result - game_pecent)
    next_me = p_me - K*(result + game_pecent)
    return next_op, next_me

# 하나의 row는 하나의 경기인 셈.
for i in range(df.shape[0]):
    op, me, result = df.iloc[i]
    student_rate[op], problem_rate[me] = changed_score(student_rate[op], problem_rate[me], result)

problem_df = pd.DataFrame.from_dict(data=problem_rate, orient='index').rename(columns={0:'problem_rate'})
student_df = pd.DataFrame.from_dict(data=student_rate, orient='index').rename(columns={0:'student_rate'})

train['student_rate'] = train.userID.apply(lambda x:student_rate[x])
train['problem_rate'] = train.assessmentItemID.apply(lambda x:problem_rate[x])

def rate(arr):  # 수능 등급 기준과 동일하게 0.5시그마 기준으로 1~9등급 나누기
    m, std = arr.mean(), arr.std()
    tarr = ((arr - m) / std + 1.75) * 2
    tarr = 9 - tarr.astype(int)
    return tarr.apply(lambda x : min(max(x,1), 9))

#학생별로 문제를 푼 수가 달라서 학생만 모아둔 df를 따로 관리하여 추가
problem_df['problem_grade'] = rate(problem_df['problem_rate'].apply(lambda x:log(x)))
student_df['student_grade'] = rate(student_df['student_rate'].apply(lambda x:log(x)))
train = pd.merge(train, problem_df, how='left', on='problem_rate')
train = pd.merge(train, student_df, how='left', on='student_rate')

#test에도 같은 칼럼 추가
# test['student_rate'] = test.userID.apply(lambda x:student_rate[x])
# test['problem_rate'] = test.assessmentItemID.apply(lambda x:problem_rate[x])
# test = pd.merge(test, problem_df, how='left', on='problem_rate')
# test = pd.merge(test, student_df, how='left', on='student_rate')
