from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.DEMO.paths_config import API_LOG_PATH
from src.DEMO.logger_config import setup_logging
from src.DEMO.running.run_single import classify_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(API_LOG_PATH)
    logger = logging.getLogger("API")
    logger.info("=" * 60)
    logger.info("Server has started successfully!")
    logger.info("Swagger UI: http://127.0.0.1:8000/docs")
    logger.info("=" * 60)
    yield


app = FastAPI(title="Appeal Processing API", lifespan=lifespan)


class AppealRequest(BaseModel):
    text: str


@app.post("/api/v1/process")
def process_appeal(request: AppealRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    try:
        summary, l2, l3, l4, fields, entities = classify_text(request.text)

        return {
            "summary": summary,
            "classification": {"L2": l2, "L3": l3, "L4": l4},
            "fields": fields,
            "entities": entities,
        }
    except Exception as e:
        logging.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Internal server error")
