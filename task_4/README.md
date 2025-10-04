# 🤖 Финальный проект по ClearML и Pytorch Lightning ⚡ https://stepik.org/lesson/1566829/step/2?unit=1587878

Структура проекта:  
.  
├── README.md      		# этот файл  
├── data  
│   ├── images     		# снимки для обучения и валидации  
│   ├── test_png   		# тестовые снимки для инференса модели  
├── docs           		# документация к датасету  
├── models         		# сохраненные модели  
│   └── best_model.ckpt		# веса модели  
├── notebooks  
│   └── EDA.ipynb  		# исследование датасета  
├── reports        		# отчеты обучения модели  
│   ├── experiment  
│   │   └── version_0  
│   │       ├── hparams.yaml  
│   │       └── metrics.csv  
│   └── inference_results.csv  
└── src                         # Модули для обучения и предсказания  
    ├── __init__.py  
    ├── .env                    # Файл с переменными окружения  
    ├── requirements.txt        # Файл с необходимыми библиотеками  
    ├── custom_classes.py       # Вспомогательные классы и модули  
    ├── make_dataset.py         # Подготовка датасета  
    ├── predict.py              # Предсказания модели  
    ├── print_time.py           # Замер времени выполнения  
    ├── set_all_seeds.py        # Установка сидов  
    └── train.py                # Обучение модели  
 
0. Установите пакеты: pip install -r requirements.txt

1. Скачиваем датасет: https://nihcc.app.box.com/v/ChestXray-NIHCC и помещаем в каталог ./data 

2. Распаковываем снимки в каталог ./data/images
  
3. Готовим описание датасета: make_dataset.py

4. Обучаем модель: train.py 

5. Результат обучения: [https://app.clear.ml/projects/b5cfcf2792744731b06ee7aa3a3b1e65/experiments/8f3c61ba79614614ad93e57f01842d8b/output/execution]

6. Вместо обучения модели скачать веса: [https://www.kaggle.com/models/saspav/finetuned-model-itsomkvit-xray-v1]

7. Инференс модели (по умолчанию тестовые снимки в каталоге ./data/test_png): predict.py
