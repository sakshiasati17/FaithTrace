from fastapi import APIRouter

from app.api.v1.endpoints import corpus, experiments, evaluation, diagnostics, recommendations, feedback

router = APIRouter()

router.include_router(corpus.router, prefix="/corpus", tags=["corpus"])
router.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
router.include_router(diagnostics.router, prefix="/diagnostics", tags=["diagnostics"])
router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
