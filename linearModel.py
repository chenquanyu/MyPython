from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


def mylinear():
    """
    线性回归预测房价
    """
    sample = load_boston()

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(sample.data, sample.target,test_size=0.2)

    # 标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1,1)).ravel()
    #y_test = std_y.transform(y_test.reshape(-1,1))

    # 正规方程
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("LR coef:", lr.coef_)

    # test collection
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("LR mean_squared_error:", mean_squared_error(y_test, y_lr_predict)) 

    # save
    path= "./lr.joblib"
    joblib.dump(lr, path)

    # load
    model = joblib.load(path)
    y_lr_predict1 = std_y.inverse_transform(model.predict(x_test))
    print("LR mean_squared_error load:", mean_squared_error(y_test, y_lr_predict1))

    # SGD 梯度下降
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("SGD coef:", sgd.coef_)

    # test collection
    y_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("SGD mean_squared_error:", mean_squared_error(y_test, y_predict)) 
    #print("SGD predict:", y_predict)
    
    # Ridge 岭回归，带有L2正则化的算法, 解决过拟合的问题
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print("Ridge coef:", rd.coef_)

    # test collection
    y_ridge_predict = std_y.inverse_transform(rd.predict(x_test))
    print("Ridge mean_squared_error:", mean_squared_error(y_test, y_ridge_predict)) 

if __name__ == "__main__":
    mylinear()
