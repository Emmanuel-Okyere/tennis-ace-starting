import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
print(df.head())
# perform exploratory analysis here:
# X = df[["FirstServe","FirstServePointsWon","FirstServeReturnPointsWon", "SecondServePointsWon", "SecondServeReturnPointsWon" , "Aces"]]
# y = df["Ranking"]
# plt.scatter(df["FirstServePointsWon"], y, alpha = 0.4)
# plt.show()

## perform single feature linear regressions here:
x = df[['FirstServeReturnPointsWon']]
y = df[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_model1_predict = model1.predict(x_test)
# print(model1.coef_)
print(model1.score(x_train, y_train))
print(model1.score(x_test, y_test))
plt.scatter(y_test,y_model1_predict)
plt.show()
plt.clf()

x = df[['BreakPointsOpportunities']]
y = df[["Winnings"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_model1_predict = model1.predict(x_test)
# print(model1.coef_)
print(model1.score(x_train, y_train))
print(model1.score(x_test, y_test))
plt.scatter(y_test,y_model1_predict)
plt.show()


## perform two feature linear regressions here:
x = df[["FirstServe","FirstServePointsWon"]]
y = df["Winnings"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
model2 = LinearRegression()
model2.fit(x_train, y_train)
y_model2_predict = model2.predict(x_test)
# print(model1.coef_)
print(model2.score(x_train, y_train))
print(model2.score(x_test, y_test))
plt.scatter(y_test,y_model2_predict)
plt.show()

# perform multiple feature linear regressions here:
x = df[["FirstServe","FirstServePointsWon","FirstServeReturnPointsWon", "SecondServePointsWon", "SecondServeReturnPointsWon" , "Aces"]]
y = df["Ranking"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
model3 = LinearRegression()
model3.fit(x_train, y_train)
y_model3_predict = model3.predict(x_test)
# print(model1.coef_)
print(model3.score(x_train, y_train))
print(model3.score(x_test, y_test))
plt.scatter(y_test,y_model3_predict)
plt.show()
