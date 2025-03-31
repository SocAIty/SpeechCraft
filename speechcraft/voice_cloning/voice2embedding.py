from io import BytesIO
from encodec.utils import convert_audio
import torchaudio
import torch

from speechcraft.core.voice_embedding import VoiceEmbedding
from speechcraft.supp.model_downloader import get_hubert_manager_and_model
from speechcraft.supp.utils import get_cpu_or_gpu


def voice2embedding(audio_file: BytesIO | str, voice_name: str = "new_speaker"):
    """
    Clones (creates and embedding of) the voice from the audio file and saves it to a .npz file.
    :param audio_file: path to the audio file or open file handle
    :param voice_name: name of the voice
    Returns: VoiceEmbedding
    """
    print("voice2embedding load models")
    hubert_manager, hubert_model, model, tokenizer = get_hubert_manager_and_model()

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_file)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)

    device = get_cpu_or_gpu()
    wav = wav.to(device)

    print("inferencing")
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    return VoiceEmbedding(codes=codes, semantic_tokens=semantic_tokens, name=voice_name)
