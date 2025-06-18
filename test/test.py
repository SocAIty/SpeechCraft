import pytest
from speechcraft import text2voice, voice2voice, voice2embedding
from media_toolkit import AudioFile
import os


# Global directory paths for use in both pytest and direct execution
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "test_files")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


@pytest.fixture
def sample_text():
    return "I love society [laughs]! [happy] What a day to make voice overs with artificial intelligence."


@pytest.fixture
def input_dir():
    return INPUT_DIR


@pytest.fixture
def output_dir():
    return OUTPUT_DIR


@pytest.fixture(autouse=True)
def setup_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def test_text2voice(sample_text, output_dir):
    audio_numpy, sample_rate = text2voice(sample_text)
    assert audio_numpy is not None

    audio = AudioFile().from_np_array(audio_numpy, sr=sample_rate)
    assert audio is not None
    output_path = f"{output_dir}/en_speaker_3_i_love_socaity.wav"

    audio.save(output_path)
    assert os.path.exists(output_path)


def test_voice2embedding(input_dir, output_dir):
    # Test voice cloning
    voice2embedding(audio_file=f"{input_dir}/voice_clone_test_voice_1.wav", voice_name="hermine").save_to_speaker_lib()

    # Test TTS with cloned voice
    tts_new_speaker, sample_rate = text2voice("Test text", voice="hermine")
    assert tts_new_speaker is not None

    audio_with_cloned_voice = AudioFile().from_np_array(tts_new_speaker, sr=sample_rate)
    assert audio_with_cloned_voice is not None
    output_path = f"{output_dir}/hermine_i_love_socaity.wav"
    audio_with_cloned_voice.save(output_path)
    assert os.path.exists(output_path)


def test_voice2voice(input_dir, output_dir):
    v2v_audio_np, sample_rate = voice2voice(audio_file=f"{input_dir}/voice_clone_test_voice_2.wav", voice_name="hermine")
    assert v2v_audio_np is not None

    v2v_audio = AudioFile().from_np_array(v2v_audio_np, sr=sample_rate)
    output_path = f"{output_dir}/potter_to_hermine.wav"
    v2v_audio.save(output_path)
    assert os.path.exists(output_path)


def test_with_media_file(input_dir, output_dir):
    """Test voice embedding creation using AudioFile object from bytes IO."""
    audio_file = AudioFile().from_file(f"{input_dir}/voice_clone_test_voice_1.wav")
    embed = voice2embedding(audio_file=audio_file.to_bytes_io(), voice_name="hermine").save_to_speaker_lib()
    assert embed is not None


if __name__ == "__main__":
    test_with_media_file()
