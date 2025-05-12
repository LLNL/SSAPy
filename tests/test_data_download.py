import os
import subprocess
from unittest import mock
import pytest

# Updated import path
from ssapy.data_utils import ensure_data_downloaded


@mock.patch("ssapy.data_utils.os.path.exists")
@mock.patch("ssapy.data_utils.os.listdir")
def test_data_already_downloaded(mock_listdir, mock_exists):
    mock_exists.return_value = True
    mock_listdir.return_value = ["somefile"]
    ensure_data_downloaded()
    mock_exists.assert_called()
    mock_listdir.assert_called()


@mock.patch("ssapy.data_utils.os.makedirs")
@mock.patch("ssapy.data_utils.subprocess.run")
@mock.patch("ssapy.data_utils.os.path.exists")
@mock.patch("ssapy.data_utils.os.listdir")
def test_download_data_when_not_present(mock_listdir, mock_exists, mock_run, mock_makedirs):
    def exists_side_effect(path):
        if "data" in path:
            return False
        if "ssapy/data" in path:
            return True
        return False

    mock_exists.side_effect = exists_side_effect
    mock_listdir.return_value = []

    ensure_data_downloaded()

    clone_call = mock.call(["git", "clone", mock.ANY, mock.ANY], check=True)
    checkout_call = mock.call(["git", "-C", mock.ANY, "checkout", mock.ANY], check=True)
    copy_call = mock.call(["cp", "-r", mock.ANY, mock.ANY], check=True)
    cleanup_call = mock.call(["rm", "-rf", mock.ANY], check=True)

    mock_run.assert_has_calls([clone_call, checkout_call, copy_call, cleanup_call], any_order=True)
    mock_makedirs.assert_called_once()


@mock.patch("ssapy.data_utils.subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd"))
@mock.patch("ssapy.data_utils.os.path.exists", return_value=False)
@mock.patch("ssapy.data_utils.os.listdir", return_value=[])
def test_git_clone_failure(mock_listdir, mock_exists, mock_run):
    with pytest.raises(RuntimeError, match="Failed to clone and copy data from GitHub."):
        ensure_data_downloaded()
