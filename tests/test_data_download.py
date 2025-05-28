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
