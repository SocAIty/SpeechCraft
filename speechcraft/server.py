import uuid
from typing import Union

from fastapi import HTTPException

from speechcraft import VoiceEmbedding

from speechcraft.settings import ALLOW_EMBEDDING_SAVE_ON_SERVER

try:
    from fast_task_api import FastTaskAPI, JobProgress, AudioFile, MediaFile
except ImportError:
    raise ImportError(
        "Please install the full version of speechcraft with pip install speechcraft[full]"
        " to use the server functionality."
    )

import speechcraft as t2v
from speechcraft.supp.model_downloader import download_all_models_init

from speechcraft.supp.utils import encode_path_safe

app = FastTaskAPI(
    title="SpeechCraft",
    summary="Create audio from text, clone voices and use them. Convert voice2voice. "
            "Generative text-to-audio Bark model.",
    version="0.0.15",
    contact={
        "name": "SocAIty",
        "url": "https://github.com/SocAIty/speechcraft",
    }
)


@app.task_endpoint(path="/text2voice")
def text2voice(
        job_progress: JobProgress,
        text: str,
        voice: Union[str, MediaFile] = "en_speaker_3",
        semantic_temp: float = 0.7,
        semantic_top_k: int = 50,
        semantic_top_p: float = 0.95,
        coarse_temp: float = 0.7,
        coarse_top_k: int = 50,
        coarse_top_p: float = 0.95,
        fine_temp: float = 0.5
):
    """
    :param text: the text to be converted
    :param voice: the name of the voice to be used. Uses the pretrained voices which are stored in models/speakers folder.
        It is also possible to provide a full path.
    :return: the audio file as bytes
    """
    # If voice is a MediaFile, it is gonna be voice cloning mode.
    if isinstance(voice, MediaFile):
        try:
            voice_name = getattr(voice, "file_name", "embedding")
            voice_name = f"{voice_name}_{uuid.uuid4()}"
            voice = VoiceEmbedding.load(voice.to_bytes_io(), speaker_name=voice_name)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid voice file. Please provide valid .npz generated with voice2voice")
    else:
        try:
            if isinstance(voice, str) and len(voice) == 0:
                print("No voice provided, using default voice")
                voice = "en_speaker_3"

            loaded_voice = VoiceEmbedding.load(voice)
            voice_name = loaded_voice.name
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid voice: {voice}")

    # validate parameters
    # remove any illegal characters from text
    message = "Started text2voice." if isinstance(voice, str) else f"Text2voice using the voice embedding {voice_name}."
    job_progress.set_status(progress=0.01, message=message)

    generated_audio_file, sample_rate = t2v.text2voice(
        text=text,
        voice=voice,
        semantic_temp=semantic_temp,
        semantic_top_k=semantic_top_k,
        semantic_top_p=semantic_top_p,
        coarse_temp=coarse_temp,
        coarse_top_k=coarse_top_k,
        coarse_top_p=coarse_top_p,
        fine_temp=fine_temp,
        progress_update_func=job_progress.set_status
    )

    # make a recognizable filename
    filename = text[:15] if len(text) > 15 else text
    filename = encode_path_safe(filename)
    filename = f"{filename}_{voice_name}.wav"
    af = AudioFile(file_name=filename).from_np_array(np_array=generated_audio_file, sr=sample_rate, file_type="wav")
    return af


@app.task_endpoint("/voice2embedding")
def voice2embedding(
        job_progress: JobProgress,
        audio_file: AudioFile,
        voice_name: str = "new_speaker",
        save: bool = ALLOW_EMBEDDING_SAVE_ON_SERVER
):
    """
    :param audio_file: the audio file as bytes 5-20s is good length
    :param voice_name: how the new voice / embedding is named
    :param save: if the embedding should be saved in the voice dir for reusage.
        Note: depending on the server settings this might not be allowed
    :return: the voice embedding as bytes
    """
    # create embedding vector
    bytesio = audio_file.to_bytes_io()
    job_progress.set_status(progress=0.1, message=f"Started embedding creation {voice_name}.")
    embedding = t2v.voice2embedding(audio_file=bytesio, voice_name=voice_name)

    # write voice embedding to file
    if save and ALLOW_EMBEDDING_SAVE_ON_SERVER:
        job_progress.set_status(progress=0.99, message=f"Saving embedding {voice_name} to library.")
        embedding.save_to_speaker_lib()

    mf = MediaFile(file_name=f"{voice_name}.npz").from_bytesio(embedding.to_bytes_io(), copy=False)
    return mf


@app.task_endpoint("/voice2voice")
def voice2voice(
        job_progress: JobProgress,
        audio_file: AudioFile,
        voice_name: str | MediaFile = "en_speaker_3",
        temp: float = 0.7
):
    """
    :param audio_file: the audio file as bytes 5-20s is good length
    :param voice_name: the new of the voice to convert to; or the voice embedding. String or MediaFile.
    :param temp: generation temperature (1.0 more diverse, 0.0 more conservative)
    :return: the converted audio file as bytes
    """
    job_progress.set_status(progress=0.01, message=f"Started voice2voice {voice_name}.")

    # inference
    audio_array, sample_rate = t2v.voice2voice(
        audio_file.to_bytes_io(), voice_name,
        temp=temp, progress_update_func=job_progress.set_status)

    job_progress.set_status(progress=0.99, message=f"Converting to audio_file {voice_name}.")

    # convert to file
    af = AudioFile(file_name=f"voice2voice_{voice_name}.wav").from_np_array(
        np_array=audio_array,
        sr=sample_rate,
        file_type="wav"
    )

    return af


def start_server(port: int = 8009):
    # first time load and install models
    download_all_models_init()
    app.start(port=port)


# start the server on provided port
if __name__ == "__main__":
    start_server()
