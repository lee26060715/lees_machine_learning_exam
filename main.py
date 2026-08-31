import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data
y = data.target

# - DataFrame 으로 만들어서 현재 경로에 `california_housing.csv` 이름의 파일로 저장
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y
df.to_csv("california_housing.csv", index=False)

# - 데이터 정보 확인 : 다음 2개 반드시 포함
#     - shape
#     - describe()
print("shape:", df.shape)
print("describe:\n", df.describe())


# - 결측치 확인
#     - 칼럼별 결측치 갯수 파악할 것
print("칼럼별 결측치 개수:\n", df[df.columns].isna().sum())

# - 데이터 탐색
#     - 가격의 분포를 히스토그램으로 확인
sns.histplot(df["target"])
plt.show()


# - 데이터 학습 준비
#     - 학습용,시험용 분리할 것
#     - `random_state` 는 42 로 고정
#     - `StandardScaler` 적용할 것
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# - 모델 학습 예측
#     - `LinearRegression` 모델 만들기
#     - 학습 데이터로 학습 시키기
#     - 결과 예측해보기
m = LinearRegression()
m.fit(X_train,y_train)
y_pred = m.predict(X_test)

# - 평가
#     - 아래 3개 지표 출력
#         - R2
#         - MAE(Mean Absolute Error)
#         - RMSE(Root Mean Squared Error)

r2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("r2:", r2)
print("MAE:", MAE)
print("RMSE:", RMSE)
