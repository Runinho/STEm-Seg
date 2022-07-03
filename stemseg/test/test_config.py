import pytest as pytest

from stemseg.config.config import YamlConfig

def test_immutability():
    cfg = YamlConfig({}, "")
    with pytest.raises(Exception):
        cfg["test"] = "somevalue"

def test_immutability_update():
    key = "test"
    value = 42
    cfg = YamlConfig({key:value}, "")

    assert cfg[key] == value

    new_value = 4711
    cfg.update_param(key, new_value)
    assert cfg[key] == new_value

