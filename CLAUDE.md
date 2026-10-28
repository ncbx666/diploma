# CLAUDE.md

Код ВКР: бенчмарк моделей прогноза благоприятных для фитофтороза картофеля условий по погодным данным (1972–2017). Рассчитан на Kaggle, работает и локально. Обученные модели используются в соседнем репозитории `diploma-web-ui`.

## Команды

- Установка: `pip install -r requirements.txt` (тяжёлые пакеты опциональны: модель пропускается, если их нет)
- Тесты: `uv run --no-project --with pandas --with numpy --with scikit-learn --with pytest python -m pytest -q`
- Smoke-прогон: `python train.py --models svm --horizons 2 --window-sizes 7 --top-n 1 --tune-trials 0 --output-dir ./outputs`

## Где что

- `train.py` — единственная точка входа бенчмарка классификаторов (данные, признаки, подбор порога, артефакты)
- `train_weather.py` — отдельный бенчмарк прогноза погодных переменных
- `src/models.py`, `src/evaluation.py` — фабрики моделей, Optuna-пространства, метрики
- `data/` — датасет, `docs/` — текст ВКР (PDF)

## Правила

- `outputs/`, `weather_outputs/` не коммитятся.
- Два теста в `tests/test_logreg_threshold.py` (`l1_ratio`, solver/penalty) падали ещё до чистки репозитория.
