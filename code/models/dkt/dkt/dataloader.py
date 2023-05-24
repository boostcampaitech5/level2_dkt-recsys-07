import os
import random
import time
from datetime import datetime
from typing import Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from math import log10 as log

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                # print(a)
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # def convert_time(s: str):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        def convert_time2timestamp(t):
            timestamp = time.mktime(t.timetuple())
            return int(timestamp)

        def convert_string2datetime(s: str):
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

        ## 문자열로 인식되는 Timestamp의 타입을 datetime으로 변경하기. 
        print('Timestamp feature engineering start ..',end='')
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
        print(' Done.')
        print('TestID feature engineering start ..',end='')
        ## 시험지의 평균 정답률, 정답 개수, 표준편차
        correct_t = df[df['userID'].shift(1) == df['userID']].groupby(['testId'])['answerCode'].agg(['mean', 'sum','std'])
        correct_t.columns = ["test_ans_mean", 'test_ans_sum','test_ans_std']
        df = pd.merge(df, correct_t, on=['testId'], how="left")

        ## 시험지의 평균 풀이 시간, 표준편차
        time_t = df.groupby(['testId'])['time'].agg(['mean', 'std'])
        time_t.columns = ["test_time_mean", 'test_time_std']
        df = pd.merge(df, time_t, on=['testId'], how="left")


        ## 시험지 대분류(test_type) 생성
        df['test_type'] = df['testId'].apply(lambda x:int(x[2]))

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
        print(' Done.')
        print('TagID feature engineering start ..',end='')
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
        print(' Done.')
        print('UserID feature engineering start ..',end='')
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

        # df['user_lvl'] = df.user_tag_lvl_mean + df.user_test_lvl_mean


        df['item_ans_1'] = df[df['userID'].shift(1) == df['userID']].groupby('assessmentItemID')['answerCode'].transform(lambda x:x.cumsum().shift(1))
        df['item_total_ans'] = df.groupby('assessmentItemID')['answerCode'].cumcount()
        df['item_acc'] = df['item_ans_1']/df['item_total_ans']
        
        print(' Done.')
        print('ItemID feature engineering start ..',end='')
        ## item의 평균 정답률, 정답 총합, 표준편차
        correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum','std'])
        correct_a.columns = ["item_ans_mean", 'item_ans_sum','item_ans_std']
        df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
        
        ## item의 평균 풀이 시간, 표준편차
        time_a = df.groupby(['assessmentItemID'])['time'].agg(['mean', 'std'])
        time_a.columns = ["item_time_mean", 'item_time_std']
        df = pd.merge(df, time_a, on=['assessmentItemID'], how="left")

        # ## item 난이도
        # df['item_lvl'] = df['item_time_mean'] / df['item_ans_mean']
        # # 범주화
        # item_cat_num = 10
        # df['item_lvl_cat'] = pd.qcut(df['item_lvl'],item_cat_num,labels=[i for i in range(item_cat_num)])

        df = df.fillna(0)
        print(' Done.')
        K=1
        ### ELO
        problems = df.assessmentItemID.unique()
        students = df.userID.unique()

        # rate는 4000으로 시작.
        problem_rate = {problem:4000 for problem in problems}
        student_rate = {student:4000 for student in students}
        elo_df = df[['userID', 'assessmentItemID', 'answerCode']]

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
        for i in range(elo_df.shape[0]):
            op, me, result = elo_df.iloc[i]
            student_rate[op], problem_rate[me] = changed_score(K,student_rate[op], problem_rate[me], result)

        problem_df = pd.DataFrame.from_dict(data=problem_rate, orient='index').rename(columns={0:'problem_rate'})
        student_df = pd.DataFrame.from_dict(data=student_rate, orient='index').rename(columns={0:'student_rate'})

        df['student_rate'] = df.userID.apply(lambda x:student_rate[x])
        df['problem_rate'] = df.assessmentItemID.apply(lambda x:problem_rate[x])
        df['log_student_rate'] = df.student_rate.apply(lambda x:log(x))

        def rate(arr):  # 수능 등급 기준과 동일하게 0.5시그마 기준으로 1~9등급 나누기. 단, 실력이 좋으면 9등급.
            m, std = arr.mean(), arr.std()
            tarr = (((arr - m) / std + 1.75) * 2).astype(int)
            return tarr.apply(lambda x : min(max(x,1), 9))

        #학생별로 문제를 푼 수가 달라서 학생만 모아둔 df를 따로 관리하여 추가
        problem_df['problem_grade'] = rate(problem_df['problem_rate'])
        student_df['student_grade'] = rate(student_df['student_rate'])
        df = pd.merge(df, problem_df, how='left', on='problem_rate')
        df = pd.merge(df, student_df, how='left', on='student_rate')

        # Elo의 log값이 실력에 선형적이므로 log를 씌운 값을 이용한다
        df['log_problem_rate'] = np.log10(df.problem_rate)

        # log값의 평균으로 df 생성
        test_grade_np = rate(df.groupby(['testId']).log_problem_rate.mean())
        test_grade_df = pd.DataFrame(test_grade_np).reset_index()
        test_grade_df.columns = ['testId', 'test_grade']
        knowledge_grade_np = rate(df.groupby(['KnowledgeTag']).log_problem_rate.mean())
        knowledge_grade_df = pd.DataFrame(knowledge_grade_np).reset_index()
        knowledge_grade_df.columns = ['KnowledgeTag', 'knowledge_grade']

        # merge
        df = df.merge(test_grade_df, how='left', on='testId')
        df = df.merge(knowledge_grade_df, how='left', on='KnowledgeTag')

        # 사용하지 않는 column 삭제
        df.drop('log_problem_rate', axis=1, inplace=True)


        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )
        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        
        # Load from data
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
        
        # Generate interaction
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
