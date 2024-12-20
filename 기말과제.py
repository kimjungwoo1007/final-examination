import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd

# 데이터 경로
train = pd.read_csv('train.csv', index_col='id')
test = pd.read_csv('test.csv', index_col='id')
submission = pd.read_csv('sample_submission.csv', index_col='id')

import numpy as np
import missingno as msno

# 훈련 데이터 복사본에서 -1을 np.NaN로 변환
train_copy = train.copy().replace(-1, np.NaN)

# 결측값 시각화(처음 28개만)
msno.bar(df=train_copy.iloc[:, 1:29], figsize=(13, 6));

def resumetable(df):
    print(f'데이터 세트 형상: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['데이터 타입'])
    summary['결측값 개수'] = (df == -1).sum().values # 피처별 -1 개수
    summary['고윳값 개수'] = df.nunique().values
    summary['데이터 종류'] = None
    for col in df.columns:
        if 'bin' in col or col == 'target':
            summary.loc[col, '데이터 종류'] = '이진형'
        elif 'cat' in col:
            summary.loc[col, '데이터 종류'] = '명목형'
        elif df[col].dtype == float:
            summary.loc[col, '데이터 종류'] = '연속형'
        elif df[col].dtype == int:
            summary.loc[col, '데이터 종류'] = '순서형'

    return summary


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib
inline


def write_percent(ax, total_size):
    '''도형 객체를 순회하며 막대 그래프 상단에 타깃값 비율 표시'''
    for patch in ax.patches:
        height = patch.get_height()  # 도형 높이(데이터 개수)
        width = patch.get_width()  # 도형 너비
        left_coord = patch.get_x()  # 도형 왼쪽 테두리의 x축 위치
        percent = height / total_size * 100  # 타깃값 비율

        # (x, y) 좌표에 텍스트 입력
        ax.text(left_coord + width / 2.0,  # x축 위치
                height + total_size * 0.001,  # y축 위치
                '{:1.1f}%'.format(percent),  # 입력 텍스트
                ha='center')  # 가운데 정렬


mpl.rc('font', size=15)
plt.figure(figsize=(7, 6))

ax = sns.countplot(x='target', data=train)
write_percent(ax, len(train))  # 비율 표시
ax.set_title('Target Distribution');

train_copy = train_copy.dropna() # np.NaN 값 삭제

plt.figure(figsize=(10, 8))
cont_corr = train_copy[cont_features].corr()     # 연속형 피처 간 상관관계
sns.heatmap(cont_corr, annot=True, cmap='OrRd'); # 히트맵 그리기