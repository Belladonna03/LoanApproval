import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_corr(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Корреляционная матрица")
    plt.show()

def plot_boxplot(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(df)
    plt.title("Boxplot для обнаружения выбросов")
    plt.show()

def analyze_column(col):
    # 1. Проверка распределения
    print(f"Анализ столбца: {col.name}")
    print("Тип данных:", col.dtype)
    print("Квантили:", col.quantile([0.25, 0.5, 0.75]))

    # 4. Визуализация распределения
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(col, kde=True)
    plt.title(f"Распределение столбца {col.name}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=col)
    plt.title(f"Boxplot столбца {col.name}")
    plt.show()
def EDA(df):
    print(df.columns)
    # print(df.info())
    #
    print(df.value_counts())
    #
    #plot_corr(df)
    #
    # df = df.drop(['person_emp_exp', 'cb_person_cred_hist_length'], axis=1)
    # print("Delete columns")
    # print(df.columns)
    for column_name in df.columns:
        column_data = df[column_name]
        df.apply(column_data)



    #plot_boxplot(df['person_income'])





filename = 'data.csv'

df = pd.read_csv(filename)

EDA(df)