from typing import Dict, Optional, Union

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic


def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    max_coarse_history: int = 300,
    output_full: bool = False,
    progress_update_func: callable = None
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        max_coarse_history: history influence. Min 60 (faster), max 630 (more context)
        progress_update_func: a callable to update the progress of the task. Called with progress float in [0, 1]
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        max_coarse_history=max_coarse_history,
        use_kv_caching=True,
        progress_update_func=progress_update_func
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath, full_generation):
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    output_full: bool = False,
    progress_update_func: callable = None
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        output_full: return full generation to be used as a history prompt
        progress_update_func: a callable to update the progress of the task.
            Called like progress_update_function(x) with x in [0, 1]
    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        output_full=output_full,
        progress_update_func=progress_update_func
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr
