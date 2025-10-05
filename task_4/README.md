# 🤖 Финальный проект по ClearML и Pytorch Lightning ⚡ https://stepik.org/lesson/1566829/step/2?unit=1587878

Задача бинарной классификации по рентгеновским снимкам: без патологий / есть патологии

Структура проекта:  
# 🗂️ Структура проекта

```bash
.
├── 📄 README.md                    # Документация проекта
├── 📁 data/                        # Данные
│   ├── 📁 images/                  # Снимки для обучения и валидации
│   └── 📁 test_png/                # Тестовые снимки для инференса
├── 📁 docs/                        # Дополнительная документация
├── 📁 models/                      # Модели машинного обучения
│   └── ⚖️ best_model.ckpt          # Веса лучшей модели
├── 📁 notebooks/                   # Jupyter ноутбуки
│   └── 🔬 EDA.ipynb                # Exploratory Data Analysis
├── 📁 reports/                     # Отчеты и результаты
│   ├── 📁 experiment/              # Эксперименты
│   │   └── 📁 version_0/           # Версия эксперимента
│   │       ├── ⚙️ hparams.yaml     # Гиперпараметры
│   │       └── 📊 metrics.csv      # Метрики обучения
│   └── 📊 inference_results.csv    # Результаты инференса
└── 📁 src/                         # Исходный код
    ├── 🐍 __init__.py              # Пакетный файл
    ├── 🔐 .env                     # Переменные окружения
    ├── 📦 requirements.txt         # Зависимости проекта
    ├── 🛠️ custom_classes.py        # Кастомные классы PyTorch
    ├── 🗃️ make_dataset.py          # Препроцессинг данных
    ├── 🔮 predict.py               # Скрипт предсказаний
    ├── ⏱️ print_time.py            # Утилиты времени
    ├── 🎲 set_all_seeds.py         # Фиксация случайных seed'ов
    └── 🏋️ train.py                 # Скрипт обучения модели
``` 
 
0. Установим пакеты: pip install -r requirements.txt

1. Скачиваем датасет: https://nihcc.app.box.com/v/ChestXray-NIHCC и помещаем в каталог ./data 

2. Распаковываем снимки в каталог ./data/images
  
3. Готовим описание датасета: make_dataset.py

4. Обучаем модель: train.py 

5. [Результат обучения](https://app.clear.ml/projects/b5cfcf2792744731b06ee7aa3a3b1e65/experiments/8f3c61ba79614614ad93e57f01842d8b/output/execution) / Вместо обучения модели [скачать веса](https://www.kaggle.com/models/saspav/finetuned-model-itsomkvit-xray-v1)

6. Инференс модели (по умолчанию тестовые снимки в каталоге ./data/test_png): predict.py
