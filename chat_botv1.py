import json
import os
import sys
import re
import logging
from typing import Optional, Dict, List
from datetime import datetime

import torch
from datasets import DatasetDict, load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, TrainerCallback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QTextEdit, QPushButton, QFileDialog,
                             QMessageBox, QTabWidget, QProgressBar, QListWidget,
                             QDockWidget, QTableWidget, QTableWidgetItem, QSpinBox,
                             QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer


class LogHandler(logging.Handler):
    """Этот класс обрабатывает вывод логов и направляет их в графический интерфейс приложения (QTextEdit),
    позволяя видеть информацию о ходе выполнения процессов."""

    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            self.text_edit.append(f"<font color='red'>{msg}</font>")
        elif record.levelno >= logging.WARNING:
            self.text_edit.append(f"<font color='orange'>{msg}</font>")
        else:
            self.text_edit.append(msg)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())


class TextEditStream:
    """Класс для перенаправления stdout/stderr в QTextEdit
    Стандартные потоки вывода перенаправляются таким образом,
    чтобы любая консольная информация отображалась прямо в окне приложения."""

    def __init__(self, text_edit, is_error=False):
        self.text_edit = text_edit
        self.is_error = is_error

    def write(self, message):
        if message.strip():
            if self.is_error:
                self.text_edit.append(f"<font color='red'>{message}</font>")
            else:
                self.text_edit.append(message)

    def flush(self):
        pass


# Настройка основного логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SystemMonitor(QObject):
    update_stats = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        import psutil
        while self.running:
            try:
                stats = {
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent,
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'status': "Работает"
                }
                self.update_stats.emit(stats)
                QThread.msleep(2000)
            except Exception as e:
                logger.error(f"Ошибка мониторинга: {str(e)}")


class TrainingProgressCallback(QObject, TrainerCallback):
    update_progress = pyqtSignal(str, str, dict)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            try:
                metrics = {
                    'epoch': f"{state.epoch:.1f}/{state.num_train_epochs}",
                    'step': f"{state.global_step}/{state.max_steps}",
                    'loss': f"{logs.get('loss', 'N/A'):.4f}",
                    'learning_rate': f"{logs.get('learning_rate', 'N/A'):.6f}",
                    'speed': f"{logs.get('speed', 'N/A')} samples/sec" if 'speed' in logs else 'N/A'
                }
                self.update_progress.emit("Обучение", "прогресс", metrics)
            except Exception as e:
                error_msg = f"Ошибка форматирования логов: {str(e)}"
                self.update_progress.emit(error_msg, "ошибка", {})


class ModelWorker(QThread):
    """
    Это фоновый рабочий поток,
     ответственный за выполнение всех трудоемких операций (загрузка модели, обучение, генерация текста и оценка).
    """
    update_signal = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int)
    complete_signal = pyqtSignal(bool, str)
    topics_loaded = pyqtSignal(list, bool)
    update_progress = pyqtSignal(str, str, dict)

    def __init__(self, operation: str, **kwargs):
        super().__init__()
        self.operation = operation
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.data_collator = None
        self.tokenized_count = 0
        self.stop_requested = False

    def run(self):
        try:
            if self.stop_requested:
                return

            if self.operation == 'load_model':
                self.load_model()
            elif self.operation == 'load_dataset':
                self.load_dataset()
                success, topics = self.analyze_dataset()
                self.topics_loaded.emit(topics if success else [], success)
            elif self.operation == 'filter_dataset':
                success, message = self.filter_dataset(self.kwargs.get('selected_topics', []))
                self.complete_signal.emit(success, message)
            elif self.operation == 'train':
                self.train_model()
            elif self.operation == 'generate':
                success, result = self.generate_text()
                self.complete_signal.emit(success, result)
            elif self.operation == 'evaluate':
                self.evaluate_model()
        except Exception as e:
            self.complete_signal.emit(False, f"Ошибка: {str(e)}")
            logger.error(f"Ошибка в ModelWorker: {str(e)}")

    def request_stop(self):
        self.stop_requested = True

    def load_model(self):
        model_path = self.kwargs.get('model_path')

        if model_path and os.path.exists(model_path):
            self.update_signal.emit(f"Загрузка модели из {model_path}", "модель")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.complete_signal.emit(True, f"Модель загружена из {model_path}")
        else:
            self.update_signal.emit("Загрузка русскоязычной модели из сети...", "модель")
            model_name = "ai-forever/rugpt3small_based_on_gpt2"
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.complete_signal.emit(True, "Русскоязычная модель загружена")

    def load_dataset(self):
        dataset_path = self.kwargs.get('dataset_path')
        self.update_signal.emit(f"Загрузка датасета из {dataset_path}", "датасет")

        try:
            if os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
                self.dataset = load_from_disk(dataset_path)
            else:
                train_path = os.path.join(dataset_path, "train")
                test_path = os.path.join(dataset_path, "test")

                if os.path.exists(train_path):
                    train_ds = load_from_disk(train_path)
                    test_ds = load_from_disk(test_path) if os.path.exists(test_path) else None

                    self.dataset = DatasetDict({
                        "train": train_ds,
                        "test": test_ds if test_ds else train_ds.select(range(1000))
                    })
                else:
                    raise ValueError("Не найдены папки train/test")

            if not isinstance(self.dataset, DatasetDict):
                raise ValueError("Загруженные данные должны быть DatasetDict")

            self.update_signal.emit("Проверка целостности данных...", "датасет")
            valid, msg = self.validate_dataset(self.dataset)
            if not valid:
                raise ValueError(f"Проблемы с данными: {msg}")
            self.update_signal.emit(msg, "датасет")

            self.dataset = self.clean_dataset(self.dataset)
            self.update_signal.emit("Датасет очищен от некорректных записей", "датасет")

            self.complete_signal.emit(True, "Датасет успешно загружен и проверен")

        except Exception as e:
            raise Exception(f"Ошибка загрузки датасета: {str(e)}")

    def validate_dataset(self, dataset):
        """Проверка целостности данных в датасете"""
        try:
            error_count = 0
            total_samples = len(dataset['train'])

            sample_size = min(100, total_samples)
            sample = dataset['train'].select(range(sample_size))

            for i, example in enumerate(sample):
                if not isinstance(example, dict):
                    error_count += 1
                    continue

                if 'conversation' not in example:
                    error_count += 1
                    continue

                dialog = example['conversation']
                if not isinstance(dialog, list) or len(dialog) == 0:
                    error_count += 1
                    continue

                last_msg = dialog[-1]
                if not isinstance(last_msg, dict) or 'content' not in last_msg:
                    error_count += 1
                    continue

            error_percent = (error_count / sample_size) * 100

            if error_percent > 5:
                return False, f"Обнаружено {error_count} ошибок в {sample_size} примерах ({error_percent:.1f}%)"

            return True, f"Проверено {sample_size} примеров, ошибок: {error_count}"

        except Exception as e:
            return False, f"Ошибка проверки данных: {str(e)}"

    def clean_dataset(self, dataset):
        """Очистка датасета от некорректных записей"""

        def is_valid_example(example):
            try:
                if not isinstance(example, dict):
                    return False
                if 'conversation' not in example:
                    return False
                dialog = example['conversation']
                if not isinstance(dialog, list) or len(dialog) == 0:
                    return False
                last_msg = dialog[-1]
                if not isinstance(last_msg, dict) or 'content' not in last_msg:
                    return False
                return True
            except:
                return False

        cleaned_dataset = dataset.filter(is_valid_example)
        removed = len(dataset['train']) - len(cleaned_dataset['train'])

        if removed > 0:
            self.update_signal.emit(
                f"Удалено {removed} некорректных записей (осталось {len(cleaned_dataset['train'])})",
                "датасет"
            )

        return cleaned_dataset

    def analyze_dataset(self):
        if not self.dataset:
            return False, "Датасет не загружен"

        try:
            topic_fields = ['topic', 'classified_topic', 'category', 'label']
            available_fields = set(self.dataset['train'].column_names)
            topic_field = next((f for f in topic_fields if f in available_fields), None)

            if not topic_field:
                return False, "В датасете нет информации о темах"

            sample = self.dataset['train'].select(range(min(1000, len(self.dataset['train']))))
            topics = list(set(sample[topic_field]))

            return True, topics

        except Exception as e:
            return False, f"Ошибка анализа тем: {str(e)}"

    def filter_dataset(self, selected_topics):
        if not self.dataset:
            return False, "Датасет не загружен"

        if not selected_topics:
            return True, "Фильтрация не применена (не выбраны темы)"

        try:
            topic_fields = ['topic', 'classified_topic', 'category', 'label']
            available_fields = set(self.dataset['train'].column_names)
            topic_field = next((f for f in topic_fields if f in available_fields), None)

            if not topic_field:
                return False, "Не найдено поле с темами для фильтрации"

            self.update_signal.emit(f"Фильтрация по темам: {', '.join(selected_topics)}", "датасет")

            def filter_function(example):
                return example[topic_field] in selected_topics

            filtered_dataset = self.dataset.filter(filter_function)
            self.dataset = filtered_dataset

            msg = (f"Фильтрация завершена\n"
                   f"Осталось примеров: {len(self.dataset['train'])}\n"
                   f"Выбранные темы: {', '.join(selected_topics)}")

            return True, msg

        except Exception as e:
            return False, f"Ошибка фильтрации: {str(e)}"

    def tokenize_function(self, examples):
        if self.stop_requested:
            raise RuntimeError("Операция отменена пользователем")

        processed = self.tokenized_count % 100
        if processed == 0:
            self.update_signal.emit(f"Токенизировано {self.tokenized_count} примеров...", "подготовка")
        self.tokenized_count += len(examples['conversation'])

        texts = []
        for dialog in examples['conversation']:
            if isinstance(dialog, list) and len(dialog) > 0:
                last_msg = dialog[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    texts.append(last_msg['content'])
                else:
                    texts.append(str(last_msg))

        if not texts:
            raise ValueError("Не удалось извлечь текст для токенизации")

        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )

    def train_model(self):
        if self.stop_requested:
            return

        if not self.model or not self.tokenizer:
            raise ValueError("Модель не загружена")
        if not self.dataset:
            raise ValueError("Датасет не загружен")

        self.tokenized_count = 0
        self.update_signal.emit("Начало подготовки данных...", "подготовка")

        try:
            sample = self.dataset['train'][0]
            sample_preview = str(sample)[:70].replace('\n', ' ') + "..."
            self.update_signal.emit(f"Пример данных: {sample_preview}", "подготовка")

            max_samples = self.kwargs.get('max_samples')
            if max_samples and max_samples > 0 and len(self.dataset['train']) > max_samples:
                self.dataset['train'] = self.dataset['train'].select(range(max_samples))
                self.update_signal.emit(f"Датасет ограничен до {max_samples} примеров", "подготовка")

            tokenized_dataset = self.dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=16,
                remove_columns=self.dataset['train'].column_names
            )

            if len(tokenized_dataset['train']) == 0:
                raise ValueError("Нет данных после токенизации")

            self.update_signal.emit(f"Токенизация завершена. Примеров: {len(tokenized_dataset['train'])}", "подготовка")

            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            training_args = TrainingArguments(
                output_dir="./model_output",
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                weight_decay=0.01,
                save_steps=200,
                logging_steps=20,
                disable_tqdm=False,
                use_cpu=True,
                remove_unused_columns=False,
                report_to=None
            )
            callback = TrainingProgressCallback()

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                data_collator=self.data_collator,
                callbacks=[callback]
            )

            self.update_signal.emit("Начало обучения модели...", "обучение")
            trainer.train()

            if self.stop_requested:
                self.update_signal.emit("Обучение остановлено пользователем", "обучение")
                return

            self.complete_signal.emit(True, "Обучение успешно завершено")

        except Exception as e:
            error_msg = f"Ошибка обучения: {str(e)}"
            self.update_signal.emit(error_msg, "ошибка")
            raise

    def generate_text(self):
        prompt = self.kwargs.get('prompt', "")
        if not prompt:
            return False, "Не указан запрос для генерации"

        try:
            gen_params = {
                'max_length': self.kwargs.get('max_length', 250),
                'temperature': self.kwargs.get('temperature', 0.7),
                'top_p': 0.95,
                'repetition_penalty': 1.5,
                'no_repeat_ngram_size': 3,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'early_stopping': True
            }

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            output = self.model.generate(input_ids, **gen_params)
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

            cleaned = self.clean_generated_text(generated, prompt)
            return True, cleaned

        except Exception as e:
            return False, f"Ошибка генерации: {str(e)}"

    def clean_generated_text(self, text, prompt):
        """
        Удаляются лишние пробелы, символы табуляции и нестандартные символы, приводится к удобочитаемому виду.
        """
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        text = ' '.join(text.split())
        text = re.sub(r'[\t•�]', '', text)
        text = ' '.join(text.split())
        text = re.sub(r'(\d+)[.)]\s*', r'\1. ', text)
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > 0:
            text = text[:last_period + 1]
        return text.strip().capitalize()

    def evaluate_model(self):
        if not self.model:
            raise ValueError("Модель не загружена")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        result = (
            f"Информация о модели:\n"
            f"Архитектура: GPT-2\n"
            f"Всего параметров: {total_params:,}\n"
            f"Обучаемых параметров: {trainable_params:,}\n"
            f"Размер словаря: {len(self.tokenizer):,}\n"
        )

        if self.dataset:
            result += f"\nИнформация о датасете:\n"
            result += f"Обучающих примеров: {len(self.dataset['train']):,}\n"
            if 'test' in self.dataset:
                result += f"Тестовых примеров: {len(self.dataset['test']):,}\n"

            sample = self.dataset['train'][0]
            result += f"\nПример данных:\n"
            for key, value in list(sample.items())[:5]:
                val_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                result += f"{key}: {val_str}\n"

        self.complete_signal.emit(True, result)


class MonitorWindow(QDockWidget):
    def __init__(self):
        super().__init__("Мониторинг системы")
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.stats_table.setRowCount(4)

        metrics = ["Загрузка CPU", "Исп. памяти", "Время", "Статус"]
        for i, metric in enumerate(metrics):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem("N/A"))

        self.stats_table.resizeColumnsToContents()
        layout.addWidget(self.stats_table)

        self.setWidget(widget)

    def update_stats(self, stats):
        self.stats_table.item(0, 1).setText(f"{stats['cpu']}%")
        self.stats_table.item(1, 1).setText(f"{stats['memory']}%")
        self.stats_table.item(2, 1).setText(stats['time'])
        self.stats_table.resizeColumnsToContents()


class NLPApplication(QMainWindow):
    """
    Представляет собой главное окно приложения, связанное с GUI-компонентами и логикой работы с моделями.
    Устанавливает начальные значения переменных, создает интерфейс и инициализирует монитор состояния системы.
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.worker = None
        self.monitor = None
        self.monitor_window = None

        self.init_ui()
        self.setWindowTitle("Обучалка Моделей нейросети")
        self.resize(1000, 800)
        self.init_monitor()
        self.setup_logging()

    def save_model(self):
        if not self.model:
            QMessageBox.warning(self, "Ошибка", "Нет модели для сохранения")
            return

        options = QFileDialog.Options()
        save_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения модели", "", options=options)

        if save_dir:
            try:
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                self.update_status(f"Модель сохранена в {save_dir}", "успех")
                QMessageBox.information(self, "Успех", "Модель успешно сохранена")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить модель: {str(e)}")

    def setup_logging(self):
        """Настраивает логгер, подключая его к графическому интерфейсу для вывода сообщений в
         специальном поле (QTextEdit) и записи логов в файл."""
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_handler = LogHandler(self.training_log)
        logger.addHandler(log_handler)

        sys.stdout = TextEditStream(self.training_log)
        sys.stderr = TextEditStream(self.training_log, is_error=True)

    def update_training_metrics(self, message, stage, metrics):
        """
        Обновляет таблицу показателей в процессе обучения модели,
        отображая прогресс обучения в режиме реального времени.
        """
        if stage == "прогресс":
            self.metrics_table.item(0, 1).setText(metrics['epoch'])
            self.metrics_table.item(1, 1).setText(metrics['step'])
            self.metrics_table.item(2, 1).setText(metrics['loss'])
            self.metrics_table.item(3, 1).setText(metrics['learning_rate'])
            if 'speed' in metrics:
                self.metrics_table.item(4, 1).setText(metrics['speed'])
            self.metrics_table.resizeColumnsToContents()

    def init_monitor(self):
        self.monitor_window = MonitorWindow()
        self.addDockWidget(Qt.RightDockWidgetArea, self.monitor_window)

        self.monitor = SystemMonitor()
        self.monitor_thread = QThread()
        self.monitor.moveToThread(self.monitor_thread)
        self.monitor.update_stats.connect(self.monitor_window.update_stats)

        self.monitor_thread.started.connect(self.monitor.run)
        self.monitor_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Вкладка чата
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)

        param_group = QWidget()
        param_layout = QHBoxLayout(param_group)

        param_layout.addWidget(QLabel("Длина ответа:"))
        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(50, 1000)
        self.max_len_spin.setValue(250)
        param_layout.addWidget(self.max_len_spin)

        param_layout.addWidget(QLabel("Температура:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setSingleStep(0.1)
        param_layout.addWidget(self.temp_spin)

        chat_layout.addWidget(param_group)

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        chat_layout.addWidget(self.chat_output)

        input_group = QWidget()
        input_layout = QHBoxLayout(input_group)

        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(80)
        self.user_input.setPlaceholderText("Введите ваш запрос (например: рецепт борща)...")
        input_layout.addWidget(self.user_input)

        self.send_btn = QPushButton("Отправить")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        chat_layout.addWidget(input_group)

        # Вкладка модели
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)

        model_group = QWidget()
        model_group_layout = QHBoxLayout(model_group)

        self.model_path_edit = QTextEdit()
        self.model_path_edit.setMaximumHeight(50)
        self.model_path_edit.setPlaceholderText("Путь к модели")
        model_group_layout.addWidget(self.model_path_edit)

        browse_model_btn = QPushButton("Обзор")
        browse_model_btn.clicked.connect(self.browse_model)
        model_group_layout.addWidget(browse_model_btn)

        load_model_btn = QPushButton("Загрузить модель")
        load_model_btn.clicked.connect(self.load_model)
        model_group_layout.addWidget(load_model_btn)

        model_layout.addWidget(model_group)

        dataset_group = QWidget()
        dataset_group_layout = QVBoxLayout(dataset_group)

        path_group = QWidget()
        path_layout = QHBoxLayout(path_group)


        self.dataset_path_edit = QTextEdit()
        self.dataset_path_edit.setMaximumHeight(50)
        self.dataset_path_edit.setPlaceholderText("Путь к датасету")
        path_layout.addWidget(self.dataset_path_edit)

        browse_dataset_btn = QPushButton("Обзор")
        browse_dataset_btn.clicked.connect(self.browse_dataset)
        path_layout.addWidget(browse_dataset_btn)

        load_dataset_btn = QPushButton("Загрузить датасет")
        load_dataset_btn.clicked.connect(self.load_dataset)
        path_layout.addWidget(load_dataset_btn)

        dataset_group_layout.addWidget(path_group)

        filter_group = QWidget()
        filter_layout = QVBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Доступные темы:"))

        self.topics_list = QListWidget()
        self.topics_list.setSelectionMode(QListWidget.MultiSelection)
        filter_layout.addWidget(self.topics_list)

        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)

        self.filter_btn = QPushButton("Фильтровать по выбранным темам")
        self.filter_btn.clicked.connect(self.apply_filter)
        btn_layout.addWidget(self.filter_btn)

        self.save_filter_btn = QPushButton("Сохранить фильтр")
        self.save_filter_btn.clicked.connect(self.save_filter)
        btn_layout.addWidget(self.save_filter_btn)

        self.load_filter_btn = QPushButton("Загрузить фильтр")
        self.load_filter_btn.clicked.connect(self.load_filter)
        btn_layout.addWidget(self.load_filter_btn)

        filter_layout.addWidget(btn_group)
        dataset_group_layout.addWidget(filter_group)

        limit_group = QWidget()
        limit_layout = QHBoxLayout(limit_group)
        limit_layout.addWidget(QLabel("Лимит примеров для обучения:"))

        self.dataset_limit = QSpinBox()
        self.dataset_limit.setRange(0, 1000000)
        self.dataset_limit.setValue(0)
        self.dataset_limit.setSpecialValueText("Без ограничения")
        limit_layout.addWidget(self.dataset_limit)

        dataset_group_layout.addWidget(limit_group)
        model_layout.addWidget(dataset_group)

        train_group = QWidget()
        train_layout = QHBoxLayout(train_group)

        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_btn)

        self.stop_train_btn = QPushButton("Остановить обучение")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        train_layout.addWidget(self.stop_train_btn)

        save_model_btn = QPushButton("Сохранить модель")
        save_model_btn.clicked.connect(self.save_model)
        train_layout.addWidget(save_model_btn)  # Добавить в layout с другими кнопками

        model_layout.addWidget(train_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self.metrics_table.setRowCount(5)
        metrics = ["Эпоха", "Шаг", "Потери", "Скорость обучения", "Скорость"]
        for i, metric in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem("N/A"))
        self.metrics_table.resizeColumnsToContents()
        model_layout.addWidget(self.metrics_table)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setPlaceholderText("Логи приложения...")
        model_layout.addWidget(self.training_log)

        eval_btn = QPushButton("Оценить модель")
        eval_btn.clicked.connect(self.evaluate_model)
        model_layout.addWidget(eval_btn)

        # Вкладка информации
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText(
            "Инструкция:\n"
            "1. Загрузите модель (локально или из сети)\n"
            "2. Загрузите датасет\n"
            "3. При необходимости отфильтруйте по темам\n"
            "4. Установите лимит примеров для обучения\n"
            "5. Обучите модель\n"
            "6. Используйте чат для генерации рецептов\n\n"
            "Советы:\n"
            "- Температура 0.7-1.0 для баланса креативности/точности\n"
            "- Длина 200-400 токенов для подробных рецептов"
        )
        info_layout.addWidget(self.info_text)

        tabs.addTab(chat_tab, "Чат")
        tabs.addTab(model_tab, "Модель")
        tabs.addTab(info_tab, "Информация")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_bar = QLabel("Готово")
        main_layout.addWidget(self.status_bar)

    def update_status(self, message, stage="инфо"):
        try:
            self.status_bar.setText(message)
            safe_message = str(message).replace('\n', ' ').replace('\r', '')

            if stage == "ошибка":
                self.training_log.append(f"<font color='red'>{safe_message}</font>")
            elif stage == "обучение":
                self.training_log.append(f"<font color='blue'>{safe_message}</font>")
            elif stage == "подготовка":
                self.training_log.append(f"<font color='green'>{safe_message}</font>")
            else:
                self.training_log.append(safe_message)

            if self.monitor_window:
                self.monitor_window.stats_table.item(3, 1).setText(safe_message[:50])

            logger.info(f"[{stage}] {safe_message}")
        except Exception as e:
            logger.error(f"Ошибка при обновлении статуса: {str(e)}")

    def browse_model(self):
        options = QFileDialog.Options()
        model_dir = QFileDialog.getExistingDirectory(
            self, "Выберите папку с моделью", "", options=options)

        if model_dir:
            self.model_path_edit.setPlainText(model_dir)

    def browse_dataset(self):
        options = QFileDialog.Options()
        dataset_dir = QFileDialog.getExistingDirectory(
            self, "Выберите папку с датасетом", "", options=options)

        if dataset_dir:
            self.dataset_path_edit.setPlainText(dataset_dir)

    def load_model(self):
        model_path = self.model_path_edit.toPlainText()

        if not model_path:
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Загрузить модель из Hugging Face Hub? (Требуется интернет)",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            model_path = None

        self.start_worker('load_model', model_path=model_path)

    def load_dataset(self):
        dataset_path = self.dataset_path_edit.toPlainText()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь к датасету")
            return

        self.topics_list.clear()
        self.start_worker('load_dataset', dataset_path=dataset_path)

        if self.worker:
            self.worker.topics_loaded.connect(self.update_topics_list)

    def update_topics_list(self, topics, success):
        self.topics_list.clear()

        if success and topics:
            self.topics_list.addItems(topics)
            self.update_status(f"Загружено {len(topics)} тем", "датасет")
            self.filter_btn.setEnabled(True)
        else:
            self.update_status("Не удалось загрузить темы или датасет не содержит информации о темах", "предупреждение")
            self.filter_btn.setEnabled(False)

    def apply_filter(self):
        selected_items = self.topics_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну тему")
            return

        selected_topics = [item.text() for item in selected_items]
        self.start_worker('filter_dataset', selected_topics=selected_topics)

    def save_filter(self):
        selected_items = self.topics_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Нет выбранных тем для сохранения")
            return

        selected_topics = [item.text() for item in selected_items]

        options = QFileDialog.Options()
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить фильтр", "", "JSON Files (*.json)", options=options)

        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(selected_topics, f, ensure_ascii=False, indent=2)
                self.update_status(f"Фильтр сохранен в {path}", "инфо")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить фильтр: {str(e)}")

    def load_filter(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить фильтр", "", "JSON Files (*.json)", options=options)

        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    selected_topics = json.load(f)

                for i in range(self.topics_list.count()):
                    self.topics_list.item(i).setSelected(False)

                for topic in selected_topics:
                    items = self.topics_list.findItems(topic, Qt.MatchExactly)
                    if items:
                        items[0].setSelected(True)

                self.update_status(f"Загружен фильтр: {', '.join(selected_topics)}", "инфо")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить фильтр: {str(e)}")

    def train_model(self):
        if not self.model:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель")
            return

        self.training_log.clear()
        self.train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)

        max_samples = self.dataset_limit.value()
        if max_samples <= 0:
            max_samples = None

        self.start_worker('train', max_samples=max_samples)

    def stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.update_status("Останавливаем обучение...", "предупреждение")
            self.train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)

    def send_message(self):
        """Реализует механизм отправки пользовательского запроса и последующей генерации ответа от модели."""
        prompt = self.user_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Ошибка", "Введите текст запроса")
            return

        if not self.model:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель")
            return

        self.chat_output.append(f"<b>Вы:</b> {prompt}")
        self.user_input.clear()

        gen_params = {
            'prompt': prompt,
            'max_length': self.max_len_spin.value(),
            'temperature': self.temp_spin.value()
        }

        self.start_worker('generate', **gen_params)

    def evaluate_model(self):
        if not self.model:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите модель")
            return

        self.start_worker('evaluate')

    def start_worker(self, operation, **kwargs):
        """Запускает фоновый поток для выполнения различных операций (загрузка модели,
        обучение, генерация текста и оценка)."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Ошибка", "Дождитесь завершения текущей операции")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = ModelWorker(operation, **kwargs)
        self.worker.update_signal.connect(self.update_status)
        self.worker.complete_signal.connect(self.operation_complete)
        self.callback = TrainingProgressCallback()
        self.worker.update_progress.connect(self.update_training_metrics)
        kwargs['callback'] = self.callback
        self.worker.kwargs = kwargs

        self.worker.model = self.model
        self.worker.tokenizer = self.tokenizer
        self.worker.dataset = self.dataset

        self.worker.start()

    def operation_complete(self, success, message):
        """Обрабатывает сигнал завершения какой-либо операции (успешное завершение или ошибку)."""
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

        if success:
            if self.worker.model:
                self.model = self.worker.model
                if self.worker.operation == 'train':
                    save_path = "./saved_model"
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    self.update_status(f"Модель сохранена в {save_path}", "успех")
            if self.worker.tokenizer:
                self.tokenizer = self.worker.tokenizer
            if self.worker.dataset:
                self.dataset = self.worker.dataset

            if self.worker.operation == 'generate':
                self.chat_output.append(f"<b>Бот:</b> {message}")
            elif self.worker.operation == 'evaluate':
                QMessageBox.information(self, "Оценка модели", message)

            self.update_status("Операция успешно завершена", "успех")
        else:
            QMessageBox.critical(self, "Ошибка", message)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(2000)

        if self.monitor:
            self.monitor.running = False
            self.monitor_thread.quit()
            self.monitor_thread.wait()

        event.accept()


if __name__ == "__main__":
    try:
        import transformers
        import datasets
        import psutil
    except ImportError as e:
        print(f"Ошибка: Не установлены требуемые библиотеки ({str(e)})")
        print("Установите: pip install transformers datasets psutil")
        sys.exit(1)

    app = QApplication(sys.argv)

    torch.set_num_threads(psutil.cpu_count(logical=False))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    window = NLPApplication()
    window.show()
    sys.exit(app.exec_())