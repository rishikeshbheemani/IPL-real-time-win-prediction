from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.services.match_replay import MatchReplay, MatchReplayError


app = FastAPI(
    title="IPL Win Prediction API",
    description="LSTM-based IPL match replay and win prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


replay = MatchReplay()


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model": "LSTM",
    }


@app.get("/api/replay/matches")
def get_matches():
    return {
        "matches": replay.list_matches()
    }


@app.get("/api/replay/{match_id}")
def get_match(match_id: str):
    try:
        match = replay.get_match(match_id)

        return {
            "match_id": match_id,
            "deliveries": len(match),
        }


    except MatchReplayError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.get(
    "/api/replay/{match_id}/delivery/{delivery_number}"
)
def get_delivery(
    match_id: str,
    delivery_number: int,
):
    try:
        return replay.get_delivery(
            match_id=match_id,
            delivery_number=delivery_number,
        )

    except MatchReplayError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

@app.get("/api/replay/{match_id}/series")
def get_replay_series(match_id: str):
    try:
        return {
            "match_id": match_id,
            "predictions": replay.get_replay_series(match_id),
        }

    except MatchReplayError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )