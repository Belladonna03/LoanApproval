import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats

# Построение и сохранение матрицы корреляции для числовых признаков
def plot_corr(df, title):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plot_filename = f"eda/{title}.png"
    plt.savefig(plot_filename)
    plt.show()

# Построение boxplot для поиска выбросов в наборе данных
def plot_boxplot(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(df)
    plt.title("Boxplot for detecting outliers")
    plt.tight_layout()
    plt.show()

# Анализ числового столбца
def analyze_int_column(column):
    # Пропуски данных
    missing_values = column.isnull().sum()

    # Основные статистики
    stats_summary = column.describe()

    # Распределение
    skewness = column.skew()
    kurtosis = column.kurtosis()

    # Выбросы (используем межквартильный размах)
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    num_outliers = len(outliers)

    # Результаты для текущего столбца
    results = {
        'Missing Values': missing_values,
        'Mean': stats_summary['mean'],
        'Median': stats_summary['50%'],
        'Std Dev': stats_summary['std'],
        'Min': stats_summary['min'],
        'Max': stats_summary['max'],
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Num Outliers': num_outliers
    }

    # Построение графиков
    plt.figure(figsize=(15, 5))

    # Гистограмма
    plt.subplot(1, 3, 1)
    sns.histplot(column.dropna(), kde=True, color='blue')
    plt.title(f'Distribution of {column.name}')

    # Boxplot
    plt.subplot(1, 3, 2)
    sns.boxplot(x=column, color='orange')
    plt.title(f'Boxplot of {column.name}')

    # Проверка на нормальность (QQ-plot)
    plt.subplot(1, 3, 3)
    stats.probplot(column.dropna(), dist="norm", plot=plt)
    plt.title(f'QQ-plot of {column.name}')
    plt.tight_layout()

    plt.savefig(f'eda/{column.name}_eda.png')
    plt.show()

    return results

# Анализ категориального столбца
def analyze_obj_column(column):
    # Пропущенные значения и уникальные значения
    missing_values = column.isnull().sum()
    unique_values = column.nunique()
    most_frequent_value = column.mode().values[0] if not column.mode().empty else None
    frequency_most_frequent = column.value_counts().max()
    relative_frequency_most_frequent = column.value_counts(normalize=True).max()
    value_counts = column.value_counts()
    relative_frequency = column.value_counts(normalize=True)
    dummies = pd.get_dummies(column, prefix=column.name)

    # Результаты для текущего столбца
    results = {
        'Missing Values': missing_values,
        'Unique Values': unique_values,
        'Most Frequent Value': most_frequent_value,
        'Frequency of Most Frequent Value': frequency_most_frequent,
        'Relative Frequency of Most Frequent Value': relative_frequency_most_frequent,
        'Value Counts': value_counts,
        'Relative Frequency': relative_frequency,
        'Dummies': dummies
    }

    # Построение круговой диаграммы
    plt.figure(figsize=(6, 6))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of "{column.name}"')
    plt.tight_layout()

    plot_filename = f"eda/{column.name}_pie_plot.png"
    plt.savefig(plot_filename)
    plt.show()

    return results

def EDA(df):
    # Общая информация о наборе данных
    print(df.info())
    print(df.describe())

    # Создание папки для сохранения графиков
    folder_name = "eda"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Построение матрицы корреляции до обработки данных
    plot_corr(df, title="Correlation_Matrix_Before_Processing")

    # Удаление ненужных столбцов
    df = df.drop(['person_emp_exp', 'cb_person_cred_hist_length'], axis=1)

    # Построение матрицы корреляции после обработки данных
    plot_corr(df, title="Correlation_Matrix_After_Processing")

    # Анализ каждого столбца
    results = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            results[col] = analyze_int_column(df[col])
        else:
            results[col] = analyze_obj_column(df[col])

    print("Column analysis results:")
    print(results)

    df = df.drop(df[df['person_age'] > 100].index)

    return df

if __name__ == "__main__":
    # Загрузка данных
    filename = 'data/data.csv'
    df = pd.read_csv(filename)

    # Проведение EDA
    df = EDA(df)

    # Сохранение обновленных данных
    df.to_csv('data/update_df.csv', index=False)