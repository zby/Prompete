import pytest
from datetime import datetime
from prompete.ReplayLLM import ReplayLiteLLM, LLMInteraction
from litellm import Message, ModelResponse, Choices, Usage
from pathlib import Path
import json
import os

@pytest.fixture
def sample_interaction():
    return {
        "timestamp": "2024-03-14T12:00:00",
        "request": {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "response": {
            "id": "test-id",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "model": "gpt-3.5-turbo",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    }

@pytest.fixture
def replay_dir(tmp_path, sample_interaction):
    # Create first interaction files
    with open(tmp_path / "1.request.json", 'w') as f:
        json.dump(sample_interaction["request"], f)
    with open(tmp_path / "1.response.json", 'w') as f:
        json.dump(sample_interaction["response"], f)
    return tmp_path

def test_replay_mode(replay_dir):
    """Test that ReplayLiteLLM correctly replays stored interactions"""
    llm = ReplayLiteLLM(replay_dir=replay_dir, replay_count=1)
    
    response = llm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hi there!"
    assert response.model == "gpt-3.5-turbo"

def test_live_mode_after_replay_exhausted(replay_dir, mocker):
    """Test that live calls are made after replay interactions are exhausted"""
    mock_response = ModelResponse(
        id="live-id",
        choices=[Choices(
            message=Message(role="assistant", content="Live response"),
            finish_reason="stop",
            index=0
        )],
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    )
    
    mocker.patch('litellm.completion', return_value=mock_response)
    
    llm = ReplayLiteLLM(replay_dir=replay_dir, replay_count=1)
    
    # First call should use replay
    response1 = llm.completion(model="gpt-3.5-turbo", messages=[])
    assert response1.choices[0].message.content == "Hi there!"
    
    # Second call should make live call
    response2 = llm.completion(model="gpt-3.5-turbo", messages=[])
    assert response2.choices[0].message.content == "Live response"

def test_save_new_interactions(tmp_path, mocker):
    """Test that new interactions are saved to file"""
    save_dir = tmp_path / "save"
    mock_response = ModelResponse(
        id="new-id",
        choices=[Choices(
            message=Message(role="assistant", content="New response"),
            finish_reason="stop",
            index=0
        )],
        model="gpt-3.5-turbo",
        usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    )
    
    mocker.patch('litellm.completion', return_value=mock_response)
    
    llm = ReplayLiteLLM(replay_dir=save_dir, save_dir=save_dir)
    llm.completion(model="gpt-3.5-turbo", messages=[])
    
    # Verify interaction was saved
    assert (save_dir / "1.request.json").exists()
    assert (save_dir / "1.response.json").exists()
    
    with open(save_dir / "1.response.json") as f:
        saved_response = json.load(f)
        assert saved_response["choices"][0]["message"]["content"] == "New response"

def test_missing_replay_directory():
    """Test that missing replay directory raises error when replay_count > 0"""
    with pytest.raises(FileNotFoundError):
        ReplayLiteLLM(replay_dir="nonexistent_dir", replay_count=1)

def test_llm_interaction_save_load(tmp_path, sample_interaction):
    """Test LLMInteraction save and load from directory"""
    interaction = LLMInteraction(**sample_interaction)
    
    # Save to directory
    interaction.save_to_directory(tmp_path, 1)
    
    # Load from directory
    loaded = LLMInteraction.load_from_directory(tmp_path, 1)
    
    assert loaded.request == sample_interaction["request"]
    assert loaded.response == sample_interaction["response"]

def test_incomplete_interaction(tmp_path, sample_interaction):
    """Test handling of incomplete interactions (missing response file)"""
    # Only save request file
    with open(tmp_path / "1.request.json", 'w') as f:
        json.dump(sample_interaction["request"], f)
    
    llm = ReplayLiteLLM(replay_dir=tmp_path, replay_count=1)
    assert len(llm.interactions) == 0  # Should skip incomplete interaction

def test_replay_dir_must_be_directory(tmp_path):
    """Test that replay_dir must be a directory, not a file"""
    # Create a file instead of a directory
    file_path = tmp_path / "not_a_dir"
    file_path.touch()
    
    with pytest.raises(ValueError, match="replay_dir must be a directory"):
        ReplayLiteLLM(replay_dir=file_path)

def test_save_dir_must_be_directory(tmp_path):
    """Test that save_dir must be a directory, not a file"""
    # Create a file instead of a directory
    file_path = tmp_path / "not_a_dir"
    file_path.touch()
    
    with pytest.raises(ValueError, match="save_dir must be a directory"):
        ReplayLiteLLM(replay_dir=tmp_path, save_dir=file_path)

def test_save_dir_created_if_not_exists(tmp_path):
    """Test that save_dir is created if it doesn't exist"""
    save_dir = tmp_path / "new_save_dir"
    assert not save_dir.exists()
    
    llm = ReplayLiteLLM(replay_dir=tmp_path, save_dir=save_dir)
    
    assert save_dir.exists()
    assert save_dir.is_dir()

def test_save_dir_cleaned_at_start(tmp_path):
    """Test that save directory is cleaned at initialization"""
    # Create save directory with some existing files
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    (save_dir / "old.request.json").write_text("{}")
    (save_dir / "old.response.json").write_text("{}")
    
    llm = ReplayLiteLLM(replay_dir=tmp_path, save_dir=save_dir)
    
    # Check that old files were removed
    assert not list(save_dir.glob("*.json"))

def test_replayed_interactions_are_saved(tmp_path, sample_interaction):
    """Test that replayed interactions are also saved to save directory"""
    # Setup replay directory with an interaction
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    with open(replay_dir / "1.request.json", 'w') as f:
        json.dump(sample_interaction["request"], f)
    with open(replay_dir / "1.response.json", 'w') as f:
        json.dump(sample_interaction["response"], f)
    
    # Setup separate save directory
    save_dir = tmp_path / "save"
    
    # Create LLM and make a replay call
    llm = ReplayLiteLLM(replay_dir=replay_dir, save_dir=save_dir, replay_count=1)
    response = llm.completion(model="gpt-3.5-turbo", messages=[])
    
    # Verify interaction was saved to save directory
    assert (save_dir / "1.request.json").exists()
    assert (save_dir / "1.response.json").exists()
    
    # Verify saved content matches original
    with open(save_dir / "1.request.json") as f:
        saved_request = json.load(f)
        assert saved_request == sample_interaction["request"]
    with open(save_dir / "1.response.json") as f:
        saved_response = json.load(f)
        assert saved_response == sample_interaction["response"]
