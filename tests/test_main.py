import main as project_main


def test_main_forwards_config_path_to_training(monkeypatch, tmp_path):
    config_path = tmp_path / "training.yaml"
    forwarded_paths = []
    monkeypatch.setattr(
        project_main,
        "train_model",
        forwarded_paths.append,
    )

    project_main.main(["--config", str(config_path)])

    assert forwarded_paths == [str(config_path)]


def test_main_uses_project_config_by_default(monkeypatch):
    forwarded_paths = []
    monkeypatch.setattr(
        project_main,
        "train_model",
        forwarded_paths.append,
    )

    project_main.main([])

    assert forwarded_paths == [str(project_main.DEFAULT_CONFIG)]
