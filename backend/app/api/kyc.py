# backend/app/api/kyc.py

from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2

from app.services.vision_pipeline import run_vision_pipeline
from app.services.similarity import check_duplicate
from app.decision.decision_engine import decide
from app.utils.logger import log

router = APIRouter(prefix="/kyc", tags=["KYC"])


def read_image(upload: UploadFile):
    """
    Convert uploaded file to OpenCV image
    """
    file_bytes = np.frombuffer(upload.file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


@router.post("/verify")
async def verify(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        log("Receiving images")

        # Convert uploads → OpenCV frames
        frame1 = read_image(image1)
        frame2 = read_image(image2)

        if frame1 is None or frame2 is None:
            return {
                "status": "rejected",
                "reason": "invalid image format"
            }

        log("Running vision pipeline")

        pipeline_result = run_vision_pipeline(frame1, frame2)

        # If pipeline fails → reject immediately
        if not pipeline_result["success"]:
            log(f"Pipeline failed: {pipeline_result['error']}")
            return {
                "status": "rejected",
                "reason": pipeline_result["error"]
            }

        embedding = pipeline_result["embedding"]

        log("Checking duplicate identity")

        duplicate = check_duplicate(embedding)

        log("Making final decision")

        decision = decide(
            pipeline_result["liveness"],
            duplicate
        )

        return decision

    except Exception as e:
        log(f"System error: {str(e)}")

        return {
            "status": "error",
            "reason": "internal processing failure"
        }

