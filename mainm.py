import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

import warnings
warnings.filterwarnings("ignore")

matches = pd.read_csv("matches.csv", index_col = 0)  #? читаем файл


matches["date"] = pd.to_datetime(matches["date"])   #? переводим все нужные нам данные базы данных в целые числа

matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex = True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")


#? готовое дерево решений
#? n_estimators - число решений дерева
#? min_samples_split - чем выше значение, тем ниже вероятность переобучения
rf = RandomForestClassifier(n_estimators = 1, min_samples_split = 10, random_state = 1)


predictors = ["venue_code", "opp_code", "hour", "day_code"]  #? массив из названий базы данных(столбцы преобразованные в числа)


def rolling_averages(group, cols, new_cols):  #? улучшенная точность алгоритма
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed = "left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)
    return group

#? gf - gols for, ga - gols against, sh - shots(броски), sot - shots on target, 
#? dist - дистанция каждого броска, pk - штрафные удары, pkatt - попытки пенальти
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]  
new_cols = [f"{c}_rolling" for c in cols]


#? применим это ко всем матчам
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
#? отбрасывает названия команд в качестве индекса, больше у нас нет нескольких индексов для базы данных
matches_rolling = matches_rolling.droplevel("team")
matches_rolling.index = range(matches_rolling.shape[0])


def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01'] #? отбор строк с определенной датой
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    prediction = precision_score(test["target"], preds)
    return combined, prediction

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
print(precision)