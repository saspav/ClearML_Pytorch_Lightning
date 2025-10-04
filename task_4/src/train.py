import torch
import warnings

from clearml import Task, Logger

from transformers import ViTImageProcessor, ViTModel

from set_all_seeds import set_all_seeds
from print_time import print_time, print_msg

from custom_classes import (SEED, CFG, check_clearml_env, parse_args,
                            CXR8DataModule, ViTBinaryClassifier, run_training,
                            )

warnings.filterwarnings('ignore', category=UserWarning)

# reproducibility
set_all_seeds(seed=SEED)

if __name__ == '__main__':

    check_clearml_env()  # читаем/устанавливаем переменные среды для clearml

    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============== Парсим аргументы командной строки =========================
    cfg = parse_args()

    cfg.model_name = "itsomk/vit-xray-v1"
    # cfg.model_name = 'google/vit-base-patch16-224-in21k'  # - оригинал

    cfg.max_epochs = 50
    # cfg.num_samples = 640

    cfg.experiment_name = f'{cfg.model_name} - samples={cfg.num_samples} - epochs={cfg.max_epochs}'

    model_name = cfg.model_name

    # Отключаем чтение предобработаннысх картинок
    cfg.images_preprocessed = None

    save_cfg = cfg.__dict__.copy()
    # Преобразуем пути Path в строки
    for param in ('root_dir', 'data_dir', 'logs_dir', 'images_dir',
                  'images_preprocessed', 'weights_path',):
        save_cfg[param] = str(save_cfg[param])

    # создаём Task (важно, чтобы он был виден глобально)
    task = Task.init(project_name=cfg.project_name,
                     task_name=cfg.experiment_name,
                     task_type=Task.TaskTypes.training)

    task.add_tags(["CXR8 Project", "Lightning", cfg.model_name])

    task.connect(save_cfg)

    logger_clearml = Logger.current_logger()

    # создаём модель
    vit_model = ViTBinaryClassifier(pretrained_model=model_name)

    # Train dataset mean/std: 0.5146206021308899 0.24966640770435333 (6000 Картинок)
    xray_mean, xray_std = [0.51462, 0.51462, 0.51462], [0.24967, 0.24967, 0.24967]

    # Загружаем препроцессор
    image_processor = ViTImageProcessor.from_pretrained(model_name,
                                                        image_mean=xray_mean,
                                                        image_std=xray_std,
                                                        )

    # print(image_processor)
    # exit()

    dm_time = print_msg('Формирование датамодуля...')
    dm = CXR8DataModule(cfg, device, processor=image_processor, sample=cfg.num_samples,
                        use_images=True)
    dm.setup()
    print_time(dm_time)

    # обучаем модель
    best_model_path = run_training(cfg, dm, vit_model, task, logger_clearml,
                                   use_class_weights=True)

    print("✅ Завершаю Task'у...")
    task.close()
