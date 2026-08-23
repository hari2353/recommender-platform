import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

REPO_DATA = ROOT.parent and (ROOT / ".." / "data").resolve()


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from recsys.api.app import create_app

    data_dir = tmp_path / "data"
    shutil.copytree(REPO_DATA, data_dir)
    return TestClient(create_app(data_dir=data_dir))


@pytest.fixture(scope="module")
def stack():
    from recsys.config import Settings
    from recsys.content.index import ContentIndex
    from recsys.engine.cmab import ThompsonBandit
    from recsys.engine.recommender import RecommenderService
    from recsys.hypotheses.manager import HypothesisManager, RoleLadder
    from recsys.scoring.profile_builder import ProfileStore

    settings = Settings()
    index = ContentIndex()
    index.load_csv(settings.data_dir / "content_catalog.csv")
    ladder = RoleLadder.load_csv(settings.data_dir / "roles.csv")
    profiles = ProfileStore()
    hypotheses = HypothesisManager(ladder, seed=3)
    bandit = ThompsonBandit(seed=9)
    service = RecommenderService(index, profiles, hypotheses, bandit=bandit)
    service.set_popularity({cid: 0.4 + (i % 10) * 0.02 for i, cid in enumerate(index.ids)})
    return settings, index, ladder, profiles, hypotheses, bandit, service
