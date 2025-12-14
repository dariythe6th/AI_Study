from __future__ import annotations

import pandas as pd
import pytest

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Фикстура для тестового DataFrame."""
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic(sample_df):
    df = sample_df
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns
    # Проверка, что добавлен новый флаг
    assert "is_categorical" in summary_df.columns


def test_missing_table_and_quality_flags(sample_df):
    df = sample_df
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories(sample_df):
    df = sample_df
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# --- НОВЫЙ ТЕСТ ДЛЯ HW03 ---

def test_compute_quality_flags_custom_heuristics():
    """
    Проверяем новые эвристики: has_constant_columns и has_high_cardinality.
    """
    df = pd.DataFrame({
        # Константная колонка (unique=1) -> True
        'constant_col': [5, 5, 5, 5, 5],
        # Высокая кардинальность (unique=6).
        # Если порог 50, то False. Если порог 5, то True.
        'high_card_col': ['a', 'b', 'c', 'd', 'e', 'f'],
        # Нормальная колонка
        'normal_col': [1, 2, 3, 4, 5, 6]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # 1. Порог кардинальности = 50 (высокая кардинальность не обнаружена)
    flags_low_threshold = compute_quality_flags(summary, missing_df, cardinality_threshold=50)
    assert flags_low_threshold['has_constant_columns'] == True, "Должен обнаружить константную колонку."
    assert flags_low_threshold['has_high_cardinality'] == False, "Не должен обнаружить высокую кардинальность (6 < 50)."

    # 2. Порог кардинальности = 5 (высокая кардинальность обнаружена)
    flags_high_threshold = compute_quality_flags(summary, missing_df, cardinality_threshold=5)
    assert flags_high_threshold['has_constant_columns'] == True, "Должен обнаружить константную колонку."
    assert flags_high_threshold['has_high_cardinality'] == True, "Должен обнаружить высокую кардинальность (6 > 5)."

    # Проверка изменения скора
    score_low = flags_low_threshold['quality_score']
    score_high = flags_high_threshold['quality_score']
    # При обнаружении высокой кардинальности скор должен упасть (на 0.1)
    # Исходный скор: 1.0 - 0.1 (за константу) = 0.9.
    # С высокой кардинальностью: 0.9 - 0.1 = 0.8.
    assert score_low > score_high
    assert score_high == pytest.approx(0.8)