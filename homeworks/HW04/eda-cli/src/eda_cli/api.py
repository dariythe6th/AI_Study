from __future__ import annotations

import io
import time
from typing import Any, Dict, Literal

import pandas as pd
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

# Импортируем ядро вашего проекта
from . import core


# ====================================================================
# Pydantic-модели
# ====================================================================

class HealthResponse(BaseModel):
    status: Literal["ok", "error"] = "ok"
    version: str = "0.1.0"

class QualityRequest(BaseModel):
    n_rows: int = Field(..., gt=0)
    n_cols: int = Field(..., gt=0)
    max_missing_share: float = Field(..., ge=0, le=1)

class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    flags: Dict[str, Any]
    latency_ms: int

class QualityFlagsResponse(BaseModel):
    flags: Dict[str, Any]
    latency_ms: int


# ====================================================================
# Утилиты
# ====================================================================

def _process_csv_and_get_flags(df: pd.DataFrame, cardinality_threshold: int) -> Dict[str, Any]:
    """
    Вспомогательная функция, которая гарантированно использует ядро EDA (HW03).
    """
    # 1. Используем summarize_dataset (Ядро)
    summary = core.summarize_dataset(df)
    
    # 2. Используем missing_table (Ядро)
    missing_df = core.missing_table(df)
    
    # 3. Используем compute_quality_flags (Ядро + доработки HW03)
    flags = core.compute_quality_flags(
        summary=summary,
        missing_df=missing_df,
        cardinality_threshold=cardinality_threshold
    )
    return flags


# ====================================================================
# API App
# ====================================================================

app = FastAPI(
    title="EDA Quality Service",
    description="HTTP API для проверки качества данных на базе eda-cli.",
    version="0.1.0",
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()

@app.post("/quality", response_model=QualityResponse)
async def simple_quality_check(request: QualityRequest):
    """Упрощенная проверка без загрузки файла."""
    start_time = time.time()
    
    summary = core.DatasetSummary(n_rows=request.n_rows, n_cols=request.n_cols, columns=[])
    missing_df = pd.DataFrame({"missing_share": [request.max_missing_share]})
    
    flags = core.compute_quality_flags(summary, missing_df)
    
    return QualityResponse(
        ok_for_model=flags["quality_score"] > 0.6,
        quality_score=flags["quality_score"],
        flags=flags,
        latency_ms=int((time.time() - start_time) * 1000)
    )

@app.post("/quality-from-csv", response_model=QualityResponse)
async def get_quality_score_from_csv(
    file: UploadFile = File(..., description="CSV файл для анализа"),
    cardinality_threshold: int = Query(50, description="Порог кардинальности из HW03")
):
    """Расчет скора и всех флагов на основе загруженного CSV."""
    start_time = time.time()

    # 1. ОБРАБОТКА ОШИБОК И ПУСТОГО CSV (Требование ревьюера)
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Разрешены только CSV файлы.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Загруженный файл пуст.")
        
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV файл не содержит данных.")
            
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении CSV: {str(exc)}")

    # 2. ИСПОЛЬЗОВАНИЕ EDA-ЯДРА (Требование ревьюера)
    flags = _process_csv_and_get_flags(df, cardinality_threshold)

    return QualityResponse(
        ok_for_model=flags["quality_score"] > 0.7,
        quality_score=flags["quality_score"],
        flags=flags,
        latency_ms=int((time.time() - start_time) * 1000)
    )

@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse)
async def get_quality_flags_from_csv(
    file: UploadFile = File(...),
    cardinality_threshold: int = Query(50)
):
    """
    Дополнительный эндпоинт (Вариант А). 
    Явно использует доработки HW03 (константные колонки, кардинальность).
    """
    start_time = time.time()

    # Повторяем логику чтения с проверкой на 400 ошибку
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Некорректный или пустой CSV.")

    # Вызов функций ядра
    flags = _process_csv_and_get_flags(df, cardinality_threshold)

    # Оставляем только флаги качества (убираем метрики)
    result_flags = {k: v for k, v in flags.items() if k not in ["quality_score", "max_missing_share"]}

    return QualityFlagsResponse(
        flags=result_flags,
        latency_ms=int((time.time() - start_time) * 1000)
    )
