import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app.main import app
from app.pipeline import RagPipeline


@pytest.fixture(scope="session")
def pipe():
    return RagPipeline()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
