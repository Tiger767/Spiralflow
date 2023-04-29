import pytest
import tempfile
from spiralflow.memory import Memory, combine_on_overlap


def test_memory_initialization():
    memory = Memory()
    assert memory.data.empty
    assert memory.index is None


def test_add_method():
    memory = Memory()
    memory.add({"text": "Hello world", "metadata": "source: test"})
    assert len(memory.data) == 1
    assert memory.index is not None
    assert memory.data.iloc[0]["text"] == "Hello world"
    assert memory.data.iloc[0]["metadata"] == "source: test"


def test_save_load_methods():
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file_path = tmp_file.name

        memory = Memory()
        memory.add({"text": "Hello world", "metadata": "source: test"})
        memory.save(tmp_file_path)

        loaded_memory = Memory(filepath=tmp_file_path)

        assert not loaded_memory.data.empty
        assert loaded_memory.index is not None
        assert loaded_memory.data.shape[0] == 1
        assert loaded_memory.data.iloc[0]["text"] == "Hello world"
        assert loaded_memory.data.iloc[0]["metadata"] == "source: test"


def test_query_method():
    memory = Memory()
    memory.add({"text": "Hello world", "metadata": "source: test"})
    memory.add({"text": "world goodbye", "metadata": "source: test"})

    result = memory.query("Hello", k=1)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world"
    assert result[0]["metadata"] == "source: test"

    result = memory.query("world", k=2)
    assert len(result) == 2

    result = memory.query("world", k=2, combine_threshold=0.3)
    assert len(result) == 1
    assert result[0]["text"] == "Hello world goodbye"


def test_combine_on_overlap():
    assert combine_on_overlap("hello world", "world peace", 0.3) == "hello world peace"
    assert combine_on_overlap("hello world", "world peace", 0.7) is None
    assert combine_on_overlap("hello world", "goodbye world", 0.1) is None


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
