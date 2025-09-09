# [Ноутбук для решения практики](https://stepik.org/lesson/1500755/step/12?unit=1520869)
# 1️⃣ **Описание шаблона для решения задачи.**
# **Задача**: обучить CatBoost, залогировать основные компоненты
# Вам необходимо сдать файл с расширением любое_имя.py в котором:
# **Базовое задание (5 баллов)**
# * Будет загрузка датасета
# * Разделение на тренировочную и валидационную выборки
# * Логирование только валидационной выборки
# * Обучение бустинга с логированием процесса обучения в ClearML и сохранением гиперпараметров модели
# * Расчет и сохранение метрики на валидационной выборке (classification report и Accuracy)
# * Сохранение обученной модели
# **Дополнительные задания (2 балла)**
# * Добавить возможность считывания 2-х параметров при запуске файла на исполнение:
#   + `-- iterations` - задаёт количество итераций бустинга (по умолчанию 500)
#   + `-- verbose`- задаёт вывод прогресса обучения CatBoost в консоль (по умолчанию False)
# Пример команды:`python любое_имя.py --iterations 200 --verbose 100`
# * Провести EDA и сохранить графики в ClearML
# 👀 При желании, рекомендуется проделать следующее:
# - Добавить теги для эксперимента
# - Добавить еще метрик и отслеживать их по мере обучения (главное в меру 😁)
# ❗️❗️❗️ **P.S.** Данный ноутбук - далеко не единственное верное решение, воспринимайте его
# как помощник для вашего собственного решения или чтобы побороть страх белого листа :)
# 2️⃣ Подключаем необходимые библиотеки

from dataclasses import dataclass, asdict

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from catboost import CatBoostClassifier
from clearml import Task, Logger
from getpass import getpass

warnings.filterwarnings('ignore', category=UserWarning, module='clearml')

# Установим опции, которые помогут привести таблицу к желаемому виду
pd.set_option('display.max_columns', None)  # реализуем возможность вывода всех столбцов
pd.set_option('display.float_format', '{:.5f}'.format)  # вывод до 5 знаков после запятой


# Парсинг аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description='CatBoost training with ClearML logging')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of boosting iterations (default: 500)')
    parser.add_argument('--verbose', type=int, default=False,
                        help='Verbose frequency (default: False, 100 for every 100 iters)')
    return parser.parse_args()


def check_clearml_env():
    required_env_vars = [
        "CLEARML_WEB_HOST",
        "CLEARML_API_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY"
    ]
    env_vars = dict(CLEARML_WEB_HOST="https://app.clear.ml/",
                    CLEARML_API_HOST="https://api.clear.ml",
                    CLEARML_FILES_HOST="https://files.clear.ml"
                    )

    for var in required_env_vars[:3]:
        if os.getenv(var) is None:
            os.environ[var] = env_vars[var]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]

    if missing_vars:
        print("⚠️  Некоторые переменные среды ClearML отсутствуют.")
        for var in missing_vars:
            os.environ[var] = getpass(f"Введите значение для {var}: ")
        print("✅ Все переменные ClearML установлены.\n")


@dataclass
class CFG:
    project_name: str = "ClearML_practice"
    experiment_name: str = "M1_ClearML_practice"

    data_path: str = "../data"
    train_name: str = "quickstart_train.csv"
    seed: int = 127


def seed_everything(seed=2024):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


check_clearml_env()

# Получаем аргументы
args = parse_args()

cfg = CFG()

seed_everything(cfg.seed)

task = Task.init(project_name=cfg.project_name,
                 task_name=cfg.experiment_name,
                 )

logger = Logger.current_logger()

# Добавьте тэги обучения
task.add_tags(["CB_classifier", "1-st Task"])

# Логируем аргументы командной строки
task.connect({"command_line_args": vars(args)})

# Конфиг запуска
task.connect(asdict(cfg), "data_config", )

# 4️⃣ Подгружаем данные
url = "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/quickstart_train.csv"
df = pd.read_csv(url)

# EDA

# Посмотрим на пропуски
missing_info = df.isna().sum()
print("Пропуски в данных:")
print(missing_info)

logger.report_table(
    title="Missing Values Analysis",
    series="Missing Count",
    table_plot=missing_info.to_frame(name='Missing Count')
)

# Распределение по видам поломок и их количеству
target_distribution = df.target_class.value_counts(normalize=True)
target_counts = df.target_class.value_counts()

# Создаем объединенную таблицу
distribution_df = pd.DataFrame({
    'Count': target_counts,
    'Proportion': target_distribution
})

print("Распределение видов поломок:")
print(distribution_df)

logger.report_table(
    title="Target Class Distribution",
    series="Complete Analysis",
    table_plot=distribution_df
)

# Распределение по видам поломок примерно одинаковое.

# Распределение по классам автомобилей
car_type_distribution = df.car_type.value_counts(normalize=True)
car_type_counts = df.car_type.value_counts()

# Создаем объединенную таблицу
car_type_df = pd.DataFrame({
    'Count': car_type_counts,
    'Proportion': car_type_distribution
})

print("Распределение по классам автомобилей:")
print(car_type_df)

# Логируем в ClearML
logger.report_table(
    title="Car Type Distribution",
    series="Complete Analysis",
    table_plot=car_type_df
)

# Основу парка автомобилей составляет эконом класс,
# а вот премиум и бизнес по 3-5% от общего количества.

# Настройка стиля графиков
sns.set(style="whitegrid")
plt.figure(figsize=(15, 5))

# 1. Гистограмма по year_to_work
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='year_to_work', bins=9, kde=True)
plt.title('Распределение по году начала работы')
plt.xlabel('Год начала работы')
plt.ylabel('Количество')

# 2. Распределение car_rating
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='car_rating', bins=30, kde=True)
plt.title('Распределение рейтинга автомобилей')
plt.xlabel('Рейтинг')
plt.ylabel('Количество')
plt.tight_layout()
# plt.show()
logger.report_matplotlib_figure(
    title="Распределение рейтинга автомобилей",
    series="Matplotlib Version",
    figure=plt.gcf(),
    report_interactive=True
)
plt.close()

# Вторая группа графиков
plt.figure(figsize=(15, 5))

# 3. Распределение riders
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='riders', bins=30, kde=True)
plt.title('Распределение количества поездок')
plt.xlabel('Число поездок')
plt.ylabel('Количество')

# 4. Распределение target_reg
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='target_reg', bins=30, kde=True)
plt.title('Распределение времени до поломки')
plt.xlabel('Время до поломки')
plt.ylabel('Количество')
plt.tight_layout()
# plt.show()
logger.report_matplotlib_figure(
    title="Распределение времени до поломки",
    series="Matplotlib Version",
    figure=plt.gcf(),
    report_interactive=True
)
plt.close()

# Распределения по годам и рейтингу сильно похожи на нормальное распределение,
# для двух других можно применить методы мат.статистики для определения вида распределения.

plt.figure(figsize=(15, 6))

sns.boxplot(
    data=df,
    y='car_type',
    x='target_reg',
    hue='car_type',  # Используем ту же переменную, что и для y
    orient='h',
    palette='viridis',
    width=0.7,
    linewidth=1.5,
    dodge=False,  # Отключаем автоматическое разделение (dodge)
    legend=False  # Отключаем легенду, так как она избыточна
)

# Добавляем заголовок и подписи
plt.title('Распределение времени до поломки по классам машин', fontsize=16, pad=20)
plt.xlabel('Время до поломки (дни)', fontsize=12)
plt.ylabel('Класс машины', fontsize=12)

# Улучшаем отображение сетки
plt.grid(axis='x', linestyle='--', alpha=0.4)

# Добавляем аннотацию с количеством наблюдений для каждого класса
for i, car_type in enumerate(df['car_type'].value_counts().index):
    count = df[df['car_type'] == car_type].shape[0]
    plt.text(
        x=df['target_reg'].max() * 0.99,
        y=i,
        s=f'N={count}',
        va='center',
        ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

plt.tight_layout()
# plt.show()
logger.report_matplotlib_figure(
    title="Распределение времени до поломки по классам машин",
    series="Matplotlib Version",
    figure=plt.gcf(),
    report_interactive=True
)
plt.close()

cat_features = ["model", "car_type", "fuel_type"]  # Выделяем категориальные признаки
targets = ["target_class", "target_reg"]
features2drop = ["car_id"]  # эти фичи будут удалены

# Отбираем итоговый набор признаков для использования моделью
filtered_features = [i for i in df.columns if (i not in targets and i not in features2drop)]
num_features = [i for i in filtered_features if i not in cat_features]

print("cat_features", cat_features)
print("num_features", len(num_features))
print("targets", targets)

# Избавляемся от NaN'ов
for c in cat_features:
    df[c] = df[c].astype(str)

train, test = train_test_split(df, test_size=0.2, random_state=cfg.seed)

# Также залогируем данные, чтобы не было путанницы в версиях датасетов
cfg.num_features = train.shape[1] - 2  # количество фичей, подаваемое на вход
cfg.num_tar_class = (train.target_class.nunique())

# Залогируйте только валидационную выборку!
print("Информация о валидационной выборке:")
print(f"Размер: {test.shape}")
print(f"Доля от общего датасета: {len(test) / len(df):.2%}")

# Основная информация о выборке
split_info = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'Samples': [len(train), len(test)],
    'Percentage': [len(train) / len(df), len(test) / len(df)]
})

logger.report_table(
    title="Dataset Split Information",
    series="Split Summary",
    table_plot=split_info
)

# Детальная информация о test выборке
test_info = pd.DataFrame({
    'Metric': ['Total Samples', 'Number of Features', 'Test Size Percentage'],
    'Value': [len(test), len(test.columns), f"{len(test) / len(df):.2%}"]
})

logger.report_table(
    title="Validation Set Details",
    series="Test Set Characteristics",
    table_plot=test_info
)

logger.report_table(title="Valid data", series="datasets", table_plot=test)

X_train = train[filtered_features]
y_train = train["target_class"]

X_test = test[filtered_features]
y_test = test["target_class"]

# 5️⃣ Обучаем модельку

cb_params = {
    "iterations": args.iterations,
    "depth": 4,
    "learning_rate": 0.06,
    "loss_function": "MultiClass",
    "custom_metric": ["Recall"],
    # Главная фишка катбуста - работа с категориальными признаками
    "cat_features": cat_features,
    # Регуляризация и ускорение
    "colsample_bylevel": 0.098,
    "subsample": 0.95,
    "l2_leaf_reg": 9,
    "min_data_in_leaf": 243,
    "max_bin": 187,
    "random_strength": 1,
    # Параметры ускорения
    "task_type": "CPU",
    "thread_count": -1,
    "bootstrap_type": "Bernoulli",
    # Важное!
    "random_seed": cfg.seed,
    "early_stopping_rounds": 50,
}

# Логирование CatBoost в ClearML https://clear.ml/docs/latest/docs/guides/frameworks/catboost/

# Логируем гиперпараметры
task.connect(cb_params)

# Замер времени обучения
start_time = time.time()

model = CatBoostClassifier(**cb_params)

# Обучение модели
model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    verbose=args.verbose,
)

training_time = time.time() - start_time
print(f"Время обучения: {training_time:.2f} секунд")

# Логируем общее время обучения
logger.report_single_value(name="Total Training Time", value=training_time)

# Предсказания и метрики на тесте
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Метрики на тесте

# Основные метрики
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_precision = precision_score(y_test, y_pred, average='weighted')
test_recall = recall_score(y_test, y_pred, average='weighted')

# ROC AUC (для многоклассовой классификации)
try:
    if len(np.unique(y_test)) > 2:
        test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:
        test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
except Exception as e:
    print(f"ROC AUC calculation skipped: {e}")
    test_auc = None

print("\n" + "=" * 50)
print("ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТЕ")
print("=" * 50)
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
if test_auc is not None:
    print(f"ROC AUC: {test_auc:.4f}")

# Логируем основные метрики
logger.report_single_value(name="Test Accuracy", value=test_accuracy)
logger.report_single_value(name="Test F1 Score", value=test_f1)
logger.report_single_value(name="Test Precision", value=test_precision)
logger.report_single_value(name="Test Recall", value=test_recall)
if test_auc is not None:
    logger.report_single_value(name="Test ROC AUC", value=test_auc)

# Логируем важность признаков
feature_importance = model.get_feature_importance()
feature_names = X_train.columns.tolist()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nТоп-10 важных признаков:")
print(importance_df.head(10))

logger.report_table(
    title="Feature Importance",
    series="All Features",
    table_plot=importance_df
)

logger.report_table(
    title="Feature Importance",
    series="Top 10 Features",
    table_plot=importance_df.head(10)
)

# Детальный отчет по классификации
cls_report = classification_report(
    y_test, y_pred, target_names=[str(x) for x in sorted(y_test.unique())], output_dict=True
)
cls_report_df = pd.DataFrame(cls_report).T

print("\nClassification Report:")
print(cls_report_df)

# 6️⃣ Сохраняем результаты в ClearML

# Логируем полный отчет классификации
logger.report_table(
    title="Classification Report",
    series="Test Set Results",
    table_plot=cls_report_df
)

# Логируем кривые обучения (если доступны)
try:
    # Логируем метрики по итерациям
    eval_metrics = model.get_evals_result()

    for metric_name, metric_values in eval_metrics['validation'].items():
        for iteration, value in enumerate(metric_values):
            logger.report_scalar(
                title="Training Metrics",
                series=metric_name,
                value=value,
                iteration=iteration
            )

except Exception as e:
    print(f"Не удалось получить метрики обучения: {e}")

# Сохраняем модель
# model.save_model('catboost_model.cbm')
task.upload_artifact('model', artifact_object='catboost_model.cbm')

print("Обучение завершено! Модель и метрики залогированы в ClearML\nЗакрываю Task'у...")

# Не забываем завершить Task'у
task.close()
