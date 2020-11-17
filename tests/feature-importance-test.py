#!pip install chefboost
from chefboost import Chefboost as cb
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv("dataset/golf.txt")
    print(df)
    print(df.head())
    config = {'algorithm': 'ID3'}
    #
    model = cb.fit(df, config)
    # #
    #Sunny,Hot,High,Weak,25
    # print(cb.predict(model, ['Sunny', 'Hot', 'High', 'Weak']))
    fi = cb.feature_importance()
    # fi.plot.bar()
    # plt.show()

