## DKT : 학생별 문제 풀이 기록 학습을 통한 특정 문제 정오답 예측

### Contributors
| [<img src="https://github.com/ji-yunkim.png" width="100px">](https://github.com/ji-yunkim) | [<img src="https://github.com/YirehEum.png" width="100px">](https://github.com/YirehEum) | [<img src="https://github.com/osmin625.png" width="100px">](https://github.com/osmin625) | [<img src="https://github.com/Grievle.png" width="100px">](https://github.com/Grievle) | [<img src="https://github.com/HannahYun.png" width="100px">](https://github.com/HannahYun) |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------:
|                          [김지연](https://github.com/ji-yunkim)                           |                            [음이레](https://github.com/YirehEum)                             |                        [오승민](https://github.com/osmin625)                           |                          [조재오](https://github.com/Grievle)                           |                            [윤한나](https://github.com/HannahYun)  

## 활용 장비 및 재료(개발 환경, 협업 tool 등)
| 항목 | 설명 |
| --- | --- |
| 환경 | • 로컬 환경: `Windows`, `Mac`<br> • 서버: `Linux (Tesla V100)`, `88GB RAM Server`<br>• 협업 Tool: `Slack`, `Notion`, `Github`<br>• 사용 버전: `Python == 3.10.11`, `Pandas == 2.0.0`, `Torch == 1.7.1`|
| Metric | AUROC Score, Accuracy Score |
| Dataset | - train/test 총합 7442명의 사용자의 학습 기록<br>- train_data.csv: 2266586개의 문항 풀이 기록<br>- test_data.csv: 260114개의 문항 풀이 기록<br>- 시계열 데이터, 총 6개의 특성<br>각 사용자의 마지막 풀이에 대한 answerCode는 -1로 기록|

## Project architecture

```
├─EDA
├─feature_engineering
├─models
│  ├─dkt
│  ├─LGBM
│  ├─lightgcn
│  ├─Transformer
│  └─Rule_based
├─postprocessing
└─preprocessing
```

## 구성

1. 시퀀스 데이터를 일반적인 지도 학습 모델로 학습하기 위한 Memory Feature 추가
2. Sequence Model 활용
3. Graph 기반 모델 활용

## 프로젝트 수행 절차 및 방법
![image](https://github.com/boostcampaitech5/level2_dkt-recsys-07/assets/46878927/d8eafa74-aee9-4c2c-8313-d650ab22fe27)

## 프로젝트 수행 결과
![image](https://github.com/boostcampaitech5/level2_dkt-recsys-07/assets/46878927/8d473d61-540d-4192-bbcb-a246df6a430a)

![image](https://github.com/boostcampaitech5/level2_dkt-recsys-07/assets/46878927/b07f9ead-f12f-469a-81e9-90ef70e40d44)

### 최종 순위
- Private 2위 (AUC: 0.8579, ACC: 0.7903)

![image](https://github.com/boostcampaitech5/level2_dkt-recsys-07/assets/46878927/3ae09afe-6480-4288-9d52-0ced0678248f)
