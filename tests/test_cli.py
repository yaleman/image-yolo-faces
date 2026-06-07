from pathlib import Path
import pytest
from image_yolo_faces.cli import cli
from click.testing import CliRunner


def test_resolve_workspaces_root_uses_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FACES_WORKSPACES_DIR", str(tmp_path / "spaces"))
    runner = CliRunner()
    result = runner.invoke(cli, ["--show-config"])
    print(f"{result.stdout=}")
    assert str((tmp_path / "spaces").resolve()) in result.stdout
