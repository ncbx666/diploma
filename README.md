# Прогноз фитофтороза картофеля по погодным данным

Код выпускной квалификационной работы: прогноз благоприятных для фитофтороза
условий (`Target_Favorable`) на `h` дней вперёд по метеоданным за
1972–2017 годы (датасет `Golitcino72-17d2`). Сравниваются классические модели, бустинги,
нейросети и статистические бейзлайны.

- Текст ВКР: [`docs/vkr-uglov.pdf`](docs/vkr-uglov.pdf)
- Веб-сервис с обученными моделями: [diploma-web-ui](https://github.com/ncbx666/diploma-web-ui),
  демо — https://diploma-web-ui.onrender.com/

Скрипты рассчитаны на Kaggle (выходные файлы по умолчанию пишутся в
`/kaggle/working`), но запускаются и локально.

## Структура

```text
train.py              # бенчмарк классификаторов болезни (основная точка входа)
train_weather.py      # бенчмарк прогноза самих погодных переменных
src/models.py         # фабрики моделей и пространства поиска Optuna
src/evaluation.py     # метрики и подбор порога
tests/                # unit-тесты подбора порога и logreg
data/Golitcino72-17d2_CLEAN.xlsx   # исходный датасет
docs/vkr-uglov.pdf    # текст ВКР
```

## Установка

```bash
pip install -r requirements.txt
```

`requirements.txt` содержит все зависимости, включая тяжёлые (`torch`,
`pytorch-forecasting`, `chronos-forecasting`). Если какой-то опциональной
библиотеки нет, соответствующая модель пропускается, а в папку результата
пишется `skipped_model.txt`.

## Данные

По умолчанию скрипты берут `data/Golitcino72-17d2_CLEAN.xlsx`. Файл ищется
также в текущей папке, в `/kaggle/working`, в подключённых Kaggle Datasets
(`/kaggle/input/**`) и по пути из `--data` или переменной
`DIPLOMA_EXCEL_FILE_PATH`.

Годы делятся по времени: последние 6 — тест, 6 перед ними — валидация,
остальные — обучение.

## Бенчмарк классификаторов: `train.py`

Быстрая проверка, что всё работает:

```bash
python train.py --models svm --horizons 2 --window-sizes 7 --top-n 1 --tune-trials 0 --output-dir ./outputs
```

Полный прогон:

```bash
python train.py --models all --horizons 2 3 --window-sizes 5 7 9 --top-n 2 --tune-trials 10
```

Модели: `logreg`, `svm`, `rf`, `gru`, `tft`, `blitecast`, `xgboost`,
`catboost`, `lightgbm`, `arima`, `sarima`; `all` запускает все.

Для каждой комбинации модель × горизонт × окно всегда прогоняются семь
вариантов признаков: `baseline`, `interaction_fe`, `temporal_fe`,
`tomek_only`, `tomek_interaction_fe`, `tomek_temporal_fe`,
`tomek_interaction_temporal_fe` (ручные взаимодействия, лаги и скользящие
статистики внутри года, Tomek Links только на обучающей выборке).

Основные флаги:

| Флаг | Что делает |
|---|---|
| `--models` | список моделей или `all` |
| `--horizons` | горизонты прогноза в днях, например `2 3` |
| `--window-sizes` | размеры окна признаков, например `5 7 9` |
| `--top-n N` | оставить N лучших результатов на модель |
| `--tune-trials N` | число итераций Optuna; `0` — без подбора |
| `--min-precision` | минимальная precision при подборе порога и ранжировании (по умолчанию 0.60) |
| `--no-y` | исключить `y1`–`y4` и производные от них признаки |
| `--invert-classes` | инвертировать итоговые предсказания |
| `--oracle`, `--predict_features` | подставлять фактическую погоду на `t+h` как «оракульные» признаки |
| `--output-dir` | папка результатов, по умолчанию `/kaggle/working/outputs` |
| `--upload-dataset`, `--dataset-slug` | сохранять zip-архивы в Kaggle Dataset |

Порог подбирается на валидации: среди порогов с precision ≥ `--min-precision`
выбирается порог с наибольшим recall. Результаты ранжируются так же: сначала
выполнение условия по precision, затем recall, PR AUC, F1 и ROC AUC.

### Результаты

Каждый эксперимент сохраняется в папку вида
`<модель>_<вариант>_<F1 в процентах>_<окно>`, например
`outputs/svm_tomek_temporal_fe_42_7/`, и сразу упаковывается в
`outputs/zips/`. В папке лежат `metrics.csv`, `predictions.csv`,
`classification_report.csv/png`, `config_used.yaml`,
`train_val_test_years.json` и графики; сводка по всем запускам — в
`outputs/summary.csv`.

Файлы в `/kaggle/working` пропадают после остановки сессии. Чтобы их
сохранить, добавьте `--upload-dataset --dataset-slug username/dataset-name`:
архивы будут скопированы в `outputs/persistent_dataset/` и выложены командой
`kaggle datasets version` (до трёх попыток).

## Бенчмарк прогноза погоды: `train_weather.py`

Прогнозирует только погодные переменные (`t_min`, `t_max`, `t_avg`,
`precipitation`, `is_rain`, `cloudiness`) на 1 и 2 дня вперёд, без целевой
переменной болезни и производных от неё признаков.

```bash
python train_weather.py --models catboost_hybrid lightgbm_hybrid --window-sizes 7 --tune-trials 0 --output-dir ./weather_outputs
python train_weather.py --models catboost_hybrid lightgbm_hybrid chronos2 --window-sizes 7 11 --tune-trials 20
```

`catboost_hybrid` и `lightgbm_hybrid` обучают по модели на каждую переменную и
смешивают её с бейзлайном persistence по валидации. `chronos2` — zero-shot
Chronos-2 (`amazon/chronos-2`, скачивается с Hugging Face при первом запуске).

Метрики: MAE, RMSE, sMAPE и R² для непрерывных переменных; accuracy, F1,
precision, recall, ROC AUC и Brier для `is_rain`. Сводка —
`summary_weather.csv`, детализация — `metrics_weather_all.csv`.

## Тесты

```bash
python -m pytest -q
```
