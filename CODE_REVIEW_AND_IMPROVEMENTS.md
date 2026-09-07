# Анализ кодовой базы, архитектурные решения и план улучшений

Данный документ содержит технический аудит кодовой базы **AI for Appeals**, анализ принятых архитектурных решений и детальный каталог проблем с рекомендациями по их устранению (в первую очередь по **путям** и **логированию**).

---

## 1. Сильные архитектурные решения

1. **Разделение пайплайна в `src/DEMO` на слои**:
   - `loading/` — загрузка тяжелых моделей, словарей и клиентов;
   - `functional/` — чистые функции бизнес-логики (классификация, NER, суммаризация, экспорт);
   - `running/` — сценарии запуска (одиночный, батчевый, UI);
   - `text_extraction/` — изолированные движки распознавания текста.
2. **Контекстный каскадный NLI-классификатор (`keryx_classifier.py`)**:
   - Вместо попытки предсказать один из 1500+ классов L4 напрямую плоской сетью, используется каскад L2 -> L3 -> L4.
   - Обогащение контекста родительскими цепочками (префиксы `"Раздел → Подтема → "`) кардинально повышает качество ранжирования кандидатов.
   - Поддержка мультилейбла через относительный порог (`score >= max_score * threshold`).
3. **Двухуровневый отказоустойчивый OCR**:
   - Приоритет отдан современной VLM (`glm-ocr`), отлично понимающей сложную верстку документов и таблиц.
   - При сбое или пустом результате срабатывает fallback на классический `PaddleOCR` + `EasyOCR` с нормализацией кириллических гомоглифов.
4. **Раздельные промпты для NER (`qwen_NER_inference.py`)**:
   - Разделение извлечения на три независимых узких промпта (персональные данные, адрес, организация-отправитель) существенно снижает галлюцинации LLM и повышает стабильность генерации JSON.
5. **Интеграция с ведомственными справочниками (`xml_export.py`)**:
   - Результаты сопоставляются с реальными справочниками СЭД (`all_refs(but_orgs).xml`, `orgs.xml`) с автоматическим получением системных `RowID`.

---

## 2. Критические проблемы и точки для улучшения

### 2.1. [РЕШЕНО] Хардкодные пути и несовместимость сред (Server vs Local)

> [!NOTE]
> **Статус**: Решено в коммите `949453d` (ветка `feature/logging-paths`).
> Создан модуль [src/DEMO/paths_config.py](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/paths_config.py) с поддержкой корня `.env` и переопределения переменных, хардкод устранен во всех файлах `src/DEMO` и `watcher.py`.

#### Где обнаружена проблема:
1. **Жестко зашитые буквы дисков и абсолютные пути Windows**:
   - [watcher.py:29](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/watcher.py#L29):
     ```python
     HOT_DIR = Path(r"D:\Обращения\0_DEMO")
     ```
     *При запуске на сервере (особенно Linux) или на машине без диска `D:` код падает с ошибкой отсутствия пути.*
   - [src/DEMO/running/run_batch.py:22](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/running/run_batch.py#L22):
     ```python
     PDF_DIR = Path(r"D:\Обращения\100_test_appeals")
     ```
   - [src/features/OCR/pdf_to_json.py:8](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/features/OCR/pdf_to_json.py#L8):
     ```python
     directory = r'D:\OCR_improving\новые выбранные'
     ```
   - [src/features/form_appeals/form_chel_appeals_w_cats.py:7](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/features/form_appeals/form_chel_appeals_w_cats.py#L7):
     ```python
     CATS_DIR = r"D:\Обращения\chel_cats"
     ```
   - [src/features/form_appeals/copy_folders.py:7-8](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/features/form_appeals/copy_folders.py#L7-L8):
     ```python
     source_dir = r"D:\Обращения\ALL"
     dest_dir = r"D:\Обращения\100_test_appeals"
     ```
   - [src/features/fileds_extraction/create_fields_dataset.py:143](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/features/fileds_extraction/create_fields_dataset.py#L143):
     ```python
     INPUT_DIR = r"D:\Обращения\xml_w_fields"
     ```
   - [src/DEMO/loading/gigachat_loader.py:11](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/loading/gigachat_loader.py#L11):
     ```python
     GIGACHAT_PATH = "../../../models/GigaChat_Lite_NEW"
     ```
2. **Дублирование вычисления `PROJECT_ROOT`**:
   - В 8 файлах строчка `PROJECT_ROOT = Path(__file__).resolve().parents[3]` повторяется независимо. Любое перемещение файла ломает путь.
3. **Почему была отменена предыдущая попытка централизации путей (коммиты `3b75ac0` -> `c308fb7`)**:
   - В файле `src/DEMO/config.py` было сделано:
     ```python
     BASE_DATA_DIR = Path(os.getenv("BASE_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
     KERYX_PATH_L2 = BASE_DATA_DIR / "models" / "KERYX_1340_G" / "L2"
     ```
     Здесь допущена ошибка: модели лежат не в `data/models`, а в корне проекта `models/`. Путь разрешался в `.../data/models/...`, чего не существовало.
   - Также в `keryx_loader.py` добавили `local_files_only=True`, что привело к падению, если huggingface не находил кэш или веса по жестко заданному пути.

#### Рекомендации по решению:
- Создать полноценный модуль конфигурации (например, [src/config.py](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/config.py) или `src/DEMO/config.py`), в котором:
  - Корень проекта определяется один раз: `BASE_DIR = Path(__file__).resolve().parents[1]` (для корня).
  - Каталоги четко разделены:
    - `DATA_DIR` (по умолчанию `BASE_DIR / "data"`, переопределяется через `DATA_DIR`)
    - `MODELS_DIR` (по умолчанию `BASE_DIR / "models"`, переопределяется через `MODELS_DIR`)
    - `HOT_FOLDER_DIR` (по умолчанию `BASE_DIR / "hot_folder"` или переменная `HOT_FOLDER_DIR`)
    - `LOGS_DIR` (по умолчанию `BASE_DIR / "logs"` или `src/DEMO/logs`)
  - Файл `.env` должен считываться из корня проекта (`find_dotenv()` или `dotenv_values()`), а не из подпапки `src/features/.env`.
  - Все пути к внешним бинарникам (`POPPLER_PATH`, `TESSERACT_CMD`, `CUDNN_PATH`) берутся через конфигуратор с валидацией.

---

### 2.2. Логирование: фрагментация, конфликты и утечки

#### Где обнаружена проблема:
1. **Конфликт `logging.basicConfig` из-за порядка импортов**:
   - В [watcher.py:8](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/watcher.py#L8) первой строчкой импортируется `from src.DEMO.running.run_single import classify_text_from_pdf`.
   - В [src/DEMO/running/run_single.py:19-26](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/running/run_single.py#L19-L26) на уровне модуля выполняется `logging.basicConfig(...)` с записью в `run_single.log`.
   - В результате, когда управление доходит до [watcher.py:18](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/watcher.py#L18), вызов `logging.basicConfig(..., LOG_FILE)` **игнорируется** стандартной библиотекой Python (так как корневой логгер уже инициализирован!). Все логи `watcher.py` отправляются в файл `run_single.log`!
2. **Относительный путь к лог-файлу в `app.py`**:
   - [src/DEMO/running/app.py:12](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/running/app.py#L12):
     ```python
     logging.FileHandler("../logs/full_pipeline.log", encoding="utf-8")
     ```
     Если запустить `python src/DEMO/running/app.py` из корня проекта, файл создастся в папке выше корня репозитория!
3. **Отсутствие логирования в `api.py`**:
   - В [api.py:40](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/api.py#L40) вызывается `logging.exception("Pipeline failed")`, но `logging.basicConfig()` не сконфигурирован — стек-трейс уходит в stderr без формата и времени.
4. **Отсутствие ротации логов**:
   - Используются простые `FileHandler`. При непрерывной работе сервиса горячей папки или API файл логов со временем займёт гигабайты и заблокирует диск.
5. **Засорение логов сырым OCR-текстом**:
   - В [run_single.py:141-143](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/running/run_single.py#L141-L143) в лог пишется весь сырой распознанный текст документа (`RAW OCR TEXT`), что раздувает логи в десятки раз.

#### Рекомендации по решению:
- Создать централизованный модуль логирования (например, `src/common/logger.py` или `src/DEMO/logging_config.py`).
- Правило: **библиотечные и функциональные модули (`functional/`, `loading/`, `running/run_single.py`) никогда не вызывают `logging.basicConfig()`!** Они лишь объявляют `logger = logging.getLogger(__name__)`.
- Только точки входа (`watcher.py`, `api.py`, `app.py`, `run_batch.py`) настраивают конфигурацию через единую функцию, например:
  ```python
  setup_logging(service_name="watcher", log_dir=LOGS_DIR, level=logging.INFO, max_bytes=20*1024*1024, backup_count=5)
  ```
- Использовать `logging.handlers.RotatingFileHandler` для автоматической ротации логов.
- Подавить шумные сторонние логгеры (`httpx`, `watchdog`, `pdfminer`, `transformers`, `urllib3`).

---

### 2.3. Загрузка моделей и аппаратные зависимости (GPU vs CPU)

#### Где обнаружена проблема:
1. **Загрузка моделей в память на этапе импорта модуля**:
   - В [src/DEMO/loading/keryx_loader.py:75-80](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/loading/keryx_loader.py#L75-L80):
     ```python
     keryx_l2 = load_keryx(KERYX_PATH_L2)
     keryx_l3 = load_keryx(KERYX_PATH_L3)
     keryx_l4 = load_keryx(KERYX_PATH_L4)
     fields_config = load_fields_config()
     field_models = load_field_models(fields_config)
     ```
     При любом импорте (например, `from src.DEMO.running.run_single import classify_text` в тестах или в API) Python сразу начинает загружать гигабайты весов нейросетей в видеопамять!
   - Если запустить скрипт на машине без CUDA или без скачанных весов, импорт модуля падает с `RuntimeError` или `FileNotFoundError`. Невозможно запустить даже `--help` или легковесные тесты.
2. **Жестко зашитый `.to("cuda")`**:
   - В [keryx_loader.py:49](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/loading/keryx_loader.py#L49), [keryx_classifier.py:32](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/functional/keryx_classifier.py#L32), [keryx_REF_classification.py:23](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/functional/keryx_REF_classification.py#L23) везде прописан `.to("cuda")`.
   - Если код запущен на сервере без GPU или локально для отладки интерфейса, он немедленно аварийно завершается.
3. **Жестко зашитый URL и модель LM Studio в `qwen_loader.py`**:
   - В [src/DEMO/loading/qwen_loader.py:8-9](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/loading/qwen_loader.py#L8-L9):
     ```python
     LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
     MODEL_NAME = "qwen/qwen3.5-9b"
     ```
     При этом в [text_extraction_glm_ocr.py:21-22](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/src/DEMO/text_extraction/text_extraction_glm_ocr.py#L21-L22) для того же LM Studio используются переменные окружения `LMSTUDIO_BASE_URL` и `LMSTUDIO_MODEL`. Возникает рассинхронизация конфигурации.

#### Рекомендации по решению:
- Сделать ленивую загрузку (Lazy loading / Singleton accessor) моделей KERYX:
  ```python
  def get_keryx_models():
      ...
  ```
  Модели должны загружаться в память только при первом вызове классификации или при старте сервиса (в `lifespan` FastAPI / `on_start` воркера).
- Определять устройство автоматически или через конфиг:
  ```python
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  ```
- Параметризовать LM Studio URL, API Key и имена моделей через переменные окружения / общий конфиг.

---

### 2.4. Архитектура сервиса горячей папки (`watcher.py`)

#### Где обнаружена проблема:
1. **Синхронная блокировка потока наблюдателя**:
   - В [watcher.py:40-56](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/watcher.py#L40-L56) функция `wait_until_file_is_ready` ждет до 30 секунд в цикле с `sleep(1)`.
   - В это время обработчик событий `FileHandler` блокирует поток `Observer`. Если в папку скопировали пачку из 20 файлов, они будут обрабатываться строго последовательно с возможными задержками и пропусками событий.
2. **Отсутствие метаданных об ошибках**:
   - При сбое файл просто переносится в `errors/`. Причина сбоя остаётся только в логе (который сейчас ещё и пишется не туда). Рядом с файлом в `errors/` целесообразно создавать `.error.txt` или `.error.json` с описанием ошибки.

#### Рекомендации по решению:
- Добавить очередь задач (`queue.Queue` или `ThreadPoolExecutor`) для асинхронной обработки файлов без блокировки потока `watchdog`.
- При перемещении в `errors/` записывать рядом лог ошибки для быстрого разбора инцидентов оператором.

---

### 2.5. REST API (`api.py`)

#### Где обнаружена проблема:
1. В [api.py:26](file:///c:/Users/Sasha/PycharmProjects/AI_for_appeals/api.py#L26):
   ```python
   @app.post("/api/v1/process")
   def process_appeal(request: AppealRequest):
   ```
   Синхронный `def` в FastAPI запускается в стандартном пуле потоков (`ThreadPoolExecutor`), но тяжелые вычисления PyTorch и долгие синхронные HTTP-запросы к LM Studio накладывают блокировки.
2. Нет эндпоинта для загрузки файла (PDF/скана). Принимается только готовый `text: str`. Для полноценного сервиса нужен эндпоинт `POST /api/v1/process-file` (`UploadFile`).
3. Нет `healthcheck` эндпоинта (`/health`, `/ready`), проверяющего доступность LM Studio и загруженность моделей KERYX.

---

## 3. Сводная таблица предлагаемых улучшений

| Приоритет | Компонент | Текущее состояние | Целевое решение |
|---|---|---|---|
| **P0** | Конфигурация путей | **[РЕШЕНО]** Устранен хардкод, создан `paths_config.py` и `.env.example` | Единый модуль конфига с автоопределением корня и поддержкой `.env` |
| **P0** | Логирование | В работе. Конфликт `basicConfig`, логи уходят в чужие файлы | Централизованный логгер, раздельные логи для `run_single`, `run_batch`, `watcher` |
| **P1** | Загрузка KERYX | Модели грузятся при импорте, жесткий `.to("cuda")` | Ленивая загрузка, автовыбор device (`cuda`/`cpu`), graceful fallback |
| **P1** | Клиент LM Studio | Разные URL в `qwen_loader` и `glm_ocr` | Единый клиент LM Studio в конфиге с проверкой связи и ретраями |
| **P2** | Сервис Watcher | Блокирующий `sleep`, перенос без метаданных | Потокобезопасная очередь, сохранение `.error.json` |
| **P2** | FastAPI (`api.py`) | Только текст, нет проверки здоровья сервиса | Добавить прием файлов PDF, healthcheck, фоновую обработку |

---

## 4. Согласованные параметры окружения
1. **ОС на удаленном сервере**: Windows 10. Запуск через `venv` напрямую из терминала.
2. **Расположение моделей**: `AI_for_appeals/models/` в корне проекта на сервере.
3. **LM Studio**: Доступен по адресу `http://127.0.0.1:1234/v1` на сервере.
4. **Интеграция с СЭД**: Пока сохраняется режим горячей папки (`watcher.py`).
