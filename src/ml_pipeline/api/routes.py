"""API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ml_pipeline.models.model import WeightsDownloadError
from ml_pipeline.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from ml_pipeline.schemas.session import SessionInfo, SessionRequest
from ml_pipeline.services.analysis import analyze_images
from ml_pipeline.services.session import Session, SessionConfigError, build_session

router = APIRouter()


def get_session(request: Request) -> Session:
    """The running session, built from the env settings on first use.

    Only one client sends requests, one at a time, so building needs no lock.

    Raises:
        HTTPException: 503 if the session cannot be built; the next request tries again.
    """
    if request.app.state.session is None:
        try:
            request.app.state.session = build_session()
        except (SessionConfigError, WeightsDownloadError) as error:
            # anticipated: the server's models are not (correctly) configured or not downloadable;
            # the client gets the reason, the log keeps the traceback
            logging.getLogger(__name__).exception("building the session failed")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return request.app.state.session


CurrentSession = Annotated[Session, Depends(get_session)]


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze(payload: AnalyzeRequest, session: CurrentSession) -> AnalyzeResponse:
    return analyze_images(payload.image_paths, session)


@router.get("/session", response_model=SessionInfo, status_code=status.HTTP_200_OK)
def read_session(session: CurrentSession) -> SessionInfo:
    """Model cards of the running models (the session is built first if there is none)."""
    return session.info()


@router.post("/session", response_model=SessionInfo, status_code=status.HTTP_200_OK)
def create_session(request: Request, payload: SessionRequest | None = None) -> SessionInfo:
    """Build a new session and replace the running one.

    Each role takes its uploaded config if given, else its env setting; without a body both come
    from env. The new session is fully built before it replaces the old one, so a failed build
    leaves the old session running.

    Raises:
        HTTPException: 422 for a missing or invalid config, 502 if weights cannot be downloaded.
    """
    if payload is None:
        detector_values, classifier_values = None, None
    else:
        detector_values, classifier_values = payload.detector, payload.classifier
    try:
        session = build_session(detector_values, classifier_values)
    except WeightsDownloadError as error:
        logging.getLogger(__name__).exception("building the requested session failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except SessionConfigError as error:
        # a client error: a warning with the traceback, not an error of the service
        logging.getLogger(__name__).warning("the requested session config is invalid", exc_info=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    request.app.state.session = session
    return session.info()
