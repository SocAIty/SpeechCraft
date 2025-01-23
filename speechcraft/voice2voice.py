from io import BytesIO

import torchaudio
from encodec.utils import convert_audio
import numpy as np

import speechcraft.supp.utils
from speechcraft.core.api import semantic_to_waveform
from speechcraft.settings import MODELS_DIR
from speechcraft.supp.model_downloader import get_hubert_manager_and_model, make_sure_models_are_downloaded


def voice2voice(
        audio_file: BytesIO | str,
        voice_name: str,
        temp: float = 0.7,
        max_coarse_history: int = 300,
        progress_update_func: callable = None
) -> tuple[np.ndarray, int]:
    """
    Takes voice and intonation from speaker_embedding and applies it to swap_audio_filename
    :param audio_file: the audio file to swap the voice. Can be a path or a file handle
    :param voice_name: the voice name or the voice embedding to use for the swap
    :param temp: generation temperature (1.0 more diverse, 0.0 more conservative)
    :param max_coarse_history: history influence. Min 60 (faster), max 630 (more context)
    :param progress_update_func: a callable to update the progress of the task.
        Called like progress_update_function(x) with x in [0, 1]
    :return:
    """
    #
    make_sure_models_are_downloaded(install_path=MODELS_DIR)
    # Load the HuBERT model
    hubert_manager, hubert_model, model, tokenizer = get_hubert_manager_and_model()

    # create a better progress function
    total_progress = 0.0
    if progress_update_func is not None:
        def progress_update_func_block(x):
            nonlocal total_progress
            prev_progress = total_progress
            curr_progress = total_progress + round(x, 2)
            if prev_progress != curr_progress:
                total_progress = curr_progress
                progress_update_func(total_progress + round(x, 2))
    else:
        progress_update_func_block = None

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_file)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    device = speechcraft.supp.utils.get_cpu_or_gpu()
    wav = wav.to(device)

    progress_update_func_block(0.05)  # 5 % for loading the audio

    # run inference
    print("embedding audio with hubert_model")
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    progress_update_func_block(0.08)  # 8 % for embedding the audio

    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    # convert voice2voice
    print("inferencing")
    output_full = False
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=voice_name,
        temp=temp,
        max_coarse_history=max_coarse_history,
        output_full=output_full,
        progress_update_func=progress_update_func_block
    )
    if output_full:
        full_generation, audio_arr = out
    else:
        audio_arr = out

    return audio_arr, model.sample_rate
