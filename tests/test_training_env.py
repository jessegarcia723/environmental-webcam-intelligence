from enviro_webcam_ml.training_env import training_environment_report


def test_training_environment_report_shape() -> None:
    report = training_environment_report()

    assert "python" in report
    assert "packages" in report
    assert "torch" in report
    assert "numpy" in report["packages"]
    assert "recommended_device" in report["torch"]
