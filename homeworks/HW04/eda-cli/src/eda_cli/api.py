from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from . import core


# ====================================================================
# Pydantic-модели для контрактов API
# ====================================================================


class HealthResponse(BaseModel):
    """Ответ на health check."""

    status: Literal["ok", "error"] = "ok"
    version: str = "0.1.0"


class QualityRequest(BaseModel):
    """Запрос для ручного расчета quality_score."""

    n_rows: int = Field(..., gt=0)
    n_cols: int = Field(..., gt=0)
    max_missing_share: float = Field(..., ge=0, le=1)


class QualityResponse(BaseModel):
    """Ответ с интегральной оценкой качества."""

    ok_for_model: bool
    quality_score: float
    flags: Dict[str, Any]  # Флаги качества, включая max_missing_share
    latency_ms: int


class QualityFlagsResponse(BaseModel):
    """
    Ответ для нового эндпоинта /quality-flags-from-csv.
    Возвращает полный набор флагов, включая эвристики HW03.
    """

    flags: Dict[str, Any]
    latency_ms: int


# ====================================================================
# Утилиты
# ====================================================================


def _load_df_from_uploadfile(file: UploadFile) -> pd.DataFrame:
    """Загружает CSV-файл из UploadFile в DataFrame."""
    try:
        # Читаем содержимое файла
        content = file.file.read()

        # Если файл пустой
        if not content:
            raise pd.errors.EmptyDataError("Uploaded file is empty")

        # Используем io.StringIO для имитации файлового объекта
        sio = io.StringIO(content.decode("utf-8"))

        # Загружаем DataFrame (автоматически определяет разделитель, если не указано)
        df = pd.read_csv(sio)

        if df.empty:
            raise pd.errors.EmptyDataError("CSV file is empty or contains only header")

        return df

    except UnicodeDecodeError as exc:
        # Ошибка кодировки
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error decoding CSV file (expected UTF-8): {exc!r}",
        ) from exc
    except pd.errors.EmptyDataError as exc:
        # Файл пустой или содержит только заголовок
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading CSV file: {exc!r}",
        ) from exc
    except Exception as exc:
        # Прочие ошибки чтения (например, битый формат)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An unexpected error occurred while reading CSV: {exc!r}",
        ) from exc


# ====================================================================
# Инициализация API
# ====================================================================

app = FastAPI(
    title="EDA Quality Service",
    description="HTTP API для проверки качества данных на базе eda-cli.",
    version="0.1.0",
)


# ====================================================================
# ЭНДПОИНТЫ ИЗ СЕМИНАРА S04
# ====================================================================

@app.get("/health", response_model=HealthResponse, name="Service Health Check")
async def health_check() -> HealthResponse:
    """Проверка работоспособности сервиса."""
    return HealthResponse()


@app.post("/quality", response_model=QualityResponse, name="Simple Quality Check")
async def simple_quality_check(
        request: QualityRequest,
        # Флаги, которые хотим проверить в ручном режиме.
        # В этом примере мы просто имитируем их, основываясь на входных данных.
        too_few_rows_threshold: int = 100,
        too_many_missing_threshold: float = 0.5,
) -> QualityResponse:
    """
    Простейшая проверка качества на основе входных параметров.
    Использует ядро eda-cli только для расчета quality_score
    по упрощенной схеме.
    """
    start_time = time.time()

    # Создаем минимальную сводку для вызова compute_quality_flags
    summary = core.DatasetSummary(
        n_rows=request.n_rows,
        n_cols=request.n_cols,
        columns=[]
    )
    # Создаем минимальную missing_df для вызова compute_quality_flags
    # Поскольку у нас нет данных по колонкам, мы используем только max_missing_share
    # и считаем, что она применима ко всем
    missing_df = pd.DataFrame({
        "missing_count": [int(request.n_rows * request.max_missing_share)],
        "missing_share": [request.max_missing_share]
    })

    # Расчет флагов и скора (используем упрощенные эвристики)
    # Важно: здесь мы не можем проверить HW03 флаги (constant/cardinality),
    # т.к. нет данных по колонкам!
    flags = core.compute_quality_flags(
        summary=summary,
        missing_df=missing_df,
        cardinality_threshold=50  # Не используется, т.к. нет колонок
    )

    # Определяем общий статус для модели
    ok_for_model = (
            not flags["too_few_rows"]
            and not flags["too_many_missing"]
            and flags["quality_score"] > 0.6  # Произвольный порог
    )

    latency_ms = int((time.time() - start_time) * 1000)

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=flags["quality_score"],
        flags=flags,
        latency_ms=latency_ms,
    )


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    status_code=status.HTTP_200_OK,
    name="Dataset Quality Score from CSV",
    description="Calculates the final quality score for a given CSV file.",
)
async def get_quality_score_from_csv(
        file: UploadFile = File(..., description="CSV file to analyze"),
        cardinality_threshold: int = Query(50,
                                           description="Cardinality threshold for 'has_high_cardinality' flag (HW03)"),
) -> QualityResponse:
    """Расчет quality_score и флагов на основе загруженного CSV."""
    start_time = time.time()

    try:
        df = _load_df_from_uploadfile(file)
    except HTTPException:
        # Пробрасываем ошибку, сгенерированную в _load_df_from_uploadfile
        raise

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)

    # Используем compute_quality_flags, который включает HW03 эвристики
    flags = core.compute_quality_flags(
        summary=summary,
        missing_df=missing_df,
        cardinality_threshold=cardinality_threshold
    )

    # Определяем общий статус для модели
    ok_for_model = (
            not flags["too_few_rows"]
            and not flags["too_many_missing"]
            and not flags["has_constant_columns"]  # Учитываем HW03 флаг
            and not flags["has_high_cardinality"]  # Учитываем HW03 флаг
            and flags["quality_score"] > 0.7  # Повышаем порог, т.к. данных больше
    )

    latency_ms = int((time.time() - start_time) * 1000)

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=flags["quality_score"],
        flags=flags,
        latency_ms=latency_ms,
    )



@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    status_code=status.HTTP_200_OK,
    name="Dataset Quality Flags from CSV",
    description="Calculates all quality flags (including HW03 custom ones) for a given CSV file.",
)
async def get_quality_flags_from_csv(
        file: UploadFile = File(..., description="CSV file to analyze"),
        # Используем параметр из HW03
        cardinality_threshold: int = Query(50,
                                           description="Cardinality threshold for 'has_high_cardinality' flag (HW03)"),
) -> QualityFlagsResponse:
    """Возвращает полный словарь флагов качества на основе загруженного CSV."""
    start_time = time.time()

    try:
        df = _load_df_from_uploadfile(file)
    except HTTPException:
        # Пробрасываем ошибку, сгенерированную в _load_df_from_uploadfile
        raise

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)

    # Вызываем compute_quality_flags, который возвращает словарь с флагами HW03
    flags = core.compute_quality_flags(
        summary=summary,
        missing_df=missing_df,
        cardinality_threshold=cardinality_threshold
    )

    latency_ms = int((time.time() - start_time) * 1000)

    flags.pop("quality_score")
    flags.pop("max_missing_share")  # max_missing_share это метрика, а не флаг
    flags.pop("cardinality_threshold")  # Порог не является флагом

    return QualityFlagsResponse(
        flags=flags,
        latency_ms=latency_ms
    )