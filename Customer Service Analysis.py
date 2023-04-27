import pandas as pd
import matplotlib.pyplot as plt
from causalinference import CausalModel

# 读取数据
df = pd.read_csv("customer_service_data.csv")

# 数据预处理
df = df.dropna()  # 删除缺失值
df = df[df["satisfaction"] >= 0]  # 删除异常值

# 建立概率模型并进行PSM处理
X = df[["training"]]  # 自变量
y = df["satisfaction"]  # 因变量
model = CausalModel(Y=y, D=X)
model.est_propensity()  # 估计倾向分数
model.trim_s()  # 进行PSM处理

# 可视化分析
matched_df = model.matched_data  # 获取匹配后的数据
plt.hist(matched_df[matched_df["training"]==0]["satisfaction"], alpha=0.5, label="No training")
plt.hist(matched_df[matched_df["training"]==1]["satisfaction"], alpha=0.5, label="Training")
plt.legend(loc="upper left")
plt.title("Distribution of satisfaction scores")
plt.xlabel("Satisfaction score")
plt.ylabel("Count")
plt.show()

# 结论
effect = model.effect_on_treated
print(f"The effect of training on satisfaction is {effect:.2f}")

