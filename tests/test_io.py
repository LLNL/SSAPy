import os

# Create a skeleton test module for ssapy.io with pytest
test_io_code = '''
import os
import tempfile
import numpy as np
import pytest
from ssapy import io

def test_exists_and_mkdir_rmdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "testdir")
        assert not io.exists(new_dir)
        io.mkdir(new_dir)
        assert io.exists(new_dir)
        io.rmdir(new_dir)
        assert not io.exists(new_dir)

def test_file_exists_extension_agnostic():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmpfile:
        fname_no_ext = os.path.splitext(tmpfile.name)[0]
    try:
        assert io.file_exists_extension_agnostic(fname_no_ext)
    finally:
        os.remove(tmpfile.name)

def test_save_and_load_pickle():
    data = {"a": 1, "b": [1, 2, 3]}
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        io.save_pickle(tmpfile.name, data)
        loaded = io.load_pickle(tmpfile.name)
        assert data == loaded
        os.remove(tmpfile.name)

def test_save_and_load_np():
    arr = np.array([1, 2, 3])
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        io.save_np(tmpfile.name, arr)
        loaded = io.load_np(tmpfile.name)
        np.testing.assert_array_equal(arr, loaded)
        os.remove(tmpfile.name)

def test_exists_in_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    with open(csv_file, 'w') as f:
        f.write("id,value\\n1,10\\n2,20\\n3,30\\n")
    assert io.exists_in_csv(str(csv_file), "id", 2)

def test_str_to_array():
    s = "[1.0, 2.0, 3.0]"
    result = io.str_to_array(s)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, expected)
'''

with open("/mnt/data/test_io.py", "w") as f:
    f.write(test_io_code)

"/mnt/data/test_io.py"

