from __future__ import annotations

import json  # <-- НОВЫЙ ИМПОРТ
from pathlib import Path
from typing import Optional, Dict, Any  # <-- НОВЫЙ ИМПОРТ

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
        path: Path,
        sep: str = ",",
        encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


def _create_json_summary(
        df: pd.DataFrame,
        summary: DatasetSummary,
        quality_flags: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Создает компактную сводку в формате JSON.
    """

    # Определяем "проблемные" колонки по наличию пропусков > 5%
    problematic_cols: List[str] = []

    for col in summary.columns:
        if col.missing_share > 0.05:
            problematic_cols.append(col.name)

    # Добавляем колонки, если обнаружены константы или высокая кардинальность
    if quality_flags.get('has_constant_columns'):
        for col in summary.columns:
            if col.unique <= 1 and col.name not in problematic_cols:
                problematic_cols.append(f"{col.name} (constant)")

    # Формируем сводку
    json_summary = {
        "dataset_info": {
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "filename": df.name if hasattr(df, 'name') else 'N/A',
        },
        "quality_score": round(quality_flags['quality_score'], 4),
        "quality_flags": {
            k: v for k, v in quality_flags.items()
            if k in ['too_few_rows', 'too_many_columns', 'too_many_missing', 'has_constant_columns',
                     'has_high_cardinality']
        },
        "problematic_columns": sorted(list(set(problematic_cols))),  # Уникальный список
        "missing_share": round(quality_flags['max_missing_share'], 4),
    }
    return json_summary


@app.command()
def overview(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
        # Основные параметры
        max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
        max_cat_columns: int = typer.Option(5, help="Максимум категориальных колонок для таблиц Top-K."),
        report_title: str = typer.Option("EDA-отчёт", help="Заголовок отчета в Markdown (H1)."),
        cardinality_threshold: int = typer.Option(50,
                                                  help="Порог уникальных значений для флага 'высокой кардинальности'."),
        # ДОПОЛНИТЕЛЬНЫЙ ПАРАМЕТР
        json_summary: bool = typer.Option(False, "--json-summary", help="Сохранить компактную сводку в summary.json."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    df.name = Path(path).name  # Временно добавляем имя файла для сводки

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, max_columns=max_cat_columns)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df, cardinality_threshold=cardinality_threshold)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. ДОПОЛНИТЕЛЬНАЯ ЧАСТЬ: JSON-сводка
    if json_summary:
        json_data = _create_json_summary(df, summary, quality_flags)
        json_path = out_root / "summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
        typer.echo(f"Компактная сводка сохранена в: {json_path}")

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        # Используем новый параметр report_title
        f.write(f"# {report_title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк (< 100): **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок (> 100): **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков (> 50%): **{quality_flags['too_many_missing']}**\n")
        # Новые флаги и параметры в отчете
        f.write(f"- **Константные колонки**: **{quality_flags['has_constant_columns']}**\n")
        f.write(
            f"- **Высокая кардинальность** (unique > {quality_flags['cardinality_threshold']}): **{quality_flags['has_high_cardinality']}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(
                f"Показаны Top-K для {len(top_cats)} из первых {max_cat_columns} категориальных колонок. См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"Сгенерированы для первых {max_hist_columns} числовых колонок. См. файлы `hist_*.png`.\n")

    # 6. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    if json_summary:
        typer.echo("- Сводка JSON: summary.json")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()