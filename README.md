  <h1 align="center" style="margin-top:-25px">SpeechCraft</h1>
<p align="center">
  <img align="center" src="docs/icon.png" height="200" />
</p>
  <h3 align="center" style="margin-top:-10px">Text2Speech, Voice-Cloning and Voice2Voice conversion</h3> 



Ever wanted to create natural sounding speech from text, clone a voice or sound like someone else? 
SpeechCraft is ideal for creating voiceovers, audiobooks, or just having fun.

# Features:
- Text2speech synthesis with the 🐶 Bark model of [Suno.ai](https://github.com/suno-ai)
  - Generate text in different languages
  - Supports emotions & singing.
- Speaker generation / embedding generation aka voice cloning 
- Voice2voice synthesis: given an audio file, generate a new audio file with the voice of a different speaker.
- Convenient deployment ready web API with [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI)
- Automatic download of models
  
# Quick Links

- [Installation](#Installation)
- [Get Started](#Inference-from-script)
- [Web Service](#Web-Service)
- [Details and guidelines](#Details-and-guidelines)

Also check-out other [socaity projects](https://github.com/orgs/SocAIty/repositories) for generative AI:
- High quality [voice2voice](https://github.com/SocAIty/Retrieval-based-Voice-Conversion-FastAPI) with retrieval based voice conversion.
- Face swapping with [Face2Face](https://github.com/SocAIty/face2face) 

## Example generations and cloned voices

https://github.com/SocAIty/SpeechCraft/assets/7961324/dbf905ea-df37-4e52-9e93-a9833352459d

The hermine voice was generated with the [voice_clone_test_voice_1.wav](https://github.com/SocAIty/SpeechCraft/tree/main/test/test_files/voice_clone_test_voice_1.wav) file with around 11 seconds of clear speech.

https://github.com/SocAIty/SpeechCraft/assets/7961324/71a039c7-e665-4576-91c7-729052e05b03


# Installation

The code runs fine on Windows and Linux. 

### With PIP
```bash
# from PyPi (without web API)
pip install speechcraft
# with web API
pip install speechcraft[full]
# or from GitHub for the newest version.
pip install git+https://github.com/SocAIty/speechcraft
```
For windows you will also need to install fairseq from a different source. 
`python 
pip install fairseq@https://github.com/Sharrnah/fairseq/releases/download/v0.12.4/fairseq-0.12.4-cp310-cp310-win_amd64.whl
`

To use a GPU don't forget to install [pytorch GPU](https://pytorch.org/get-started/locally/) with your correct 
[cuda](https://developer.nvidia.com/cuda-downloads) version.
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`


### Or clone and work with the repository.
1. Clone the repository.
2. (Optional) Create a virtual environment. With `python -m venv venv` and activate it with `venv/Scripts/activate`.
3. Install `pip install .`
4. Don't forget to install fairseq and pytorch GPU.



# Usage
We provide three ways to use the face swapping functionality.
1. [Direct module import and inference](#Inference-from-script) 
2. [By deploying and calling the web service](#Web-Service)
3. As part of the [socaity SDK](https://github.com/SocAIty/socaity).  # coming soon


## Inference from script

```python
from speechcraft import text2voice, voice2embedding, voice2voice

# simple text2speech synthesis
text = "I love society [laughs]! [happy] What a day to make voice overs with artificial intelligence."
audio_numpy, sample_rate = text2voice(text, speaker_name="en_speaker_3")

# speaker embedding generation
embedding = voice2embedding(audio_file="voice_sample_15s.wav", voice_name="hermine").save_to_speaker_lib()

# text2speech synthesis with cloned voice or embedding
audio_with_cloned_voice, sample_rate = text2voice(sample_text, voice=embedding)  # also works with voice="hermine"

# voice2voice synthesis
cloned_audio = voice2voice(audio_file="my_audio_file.wav", voice_name_or_embedding_path="hermine")

```
Use the following code to convert and save the audio file with the [media-toolkit](https://github.com/SocAIty/media-toolkit) module.
```python
from media_toolkit import AudioFile
audio = AudioFile().from_np_array(audio_numpy, sr=sample_rate)
audio.save("my_new_audio.wav")
```

Note: The first time your are using speechcraft it will download the models.
These files are quite big and can take a while to download.


## Web Service

![image of openapi server](docs/server_screenshot.png)

The usage of the webservice is documented in [WebService.md](docs/WebService.md).

# Details and guidelines

## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).

On enterprise GPUs and PyTorch nightly, Bark can generate audio in roughly real-time. On older GPUs, default colab, or CPU, inference time might be significantly slower. For older GPUs or CPU you might want to consider using smaller models. Details can be found in out tutorial sections here.

The full version of Bark requires around 12GB of VRAM to hold everything on GPU at the same time. 
To use a smaller version of the models, which should fit into 8GB VRAM, set the environment flag `SUNO_USE_SMALL_MODELS=True`.

If you don't have hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our model playground [here](https://suno-ai.typeform.com/suno-studio).

## ⚙️ Emotion and language with the Model

Bark is fully generative text-to-audio model devolved for research and demo purposes. It follows a GPT style architecture similar to [AudioLM](https://arxiv.org/abs/2209.03143) and [Vall-E](https://arxiv.org/abs/2301.02111) and a quantized Audio representation from [EnCodec](https://github.com/facebookresearch/encodec). It is not a conventional TTS model, but instead a fully generative text-to-audio model capable of deviating in unexpected ways from any given script. Different to previous approaches, the input text prompt is converted directly to audio without the intermediate use of phonemes. It can therefore generalize to arbitrary instructions beyond speech such as music lyrics, sound effects or other non-speech sounds.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://suno.ai/discord)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

### Supported Languages

| Language | Status |
| --- | :---: |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |

To use a different language use the corresponding voice parameter to it like "de_speaker_1".
You find preset voices and languages in the assets folder.

## © License

SpeechCraft and Bark is licensed under the MIT License. 


## Voice Cloning guide

Make sure these things are NOT in your voice input: (in no particular order)

- Noise (You can use a noise remover before)
- Music (There are also music remover tools) (Unless you want music in the background)
- A cut-off at the end (This will cause it to try and continue on the generation)

What makes for good prompt audio? (in no particular order)

- Around ~7 to ~15 seconds of voice data
- Clearly spoken
- No weird background noises
- Only one speaker
- Audio which ends after a sentence ends
- Regular/common voice (They usually have more success, it's still capable of cloning complex voices, but not as good at it)


# Disclaimer
This repository is a merge of the orignal [bark repository](https://github.com/suno-ai/bark) and [bark-voice-cloning-HuBert-quantizer](https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer/blob/master/readme.md) by [gitmylo](https://github.com/gitmylo)
The credit goes to the original authors. Like the original authors, I am also not responsible for any misuse of this repository. Use at your own risk, and please act responsibly.
Don't copy and publish the voice of a person without their consent.

# Contribute

Any help with maintaining and extending the package is welcome. Feel free to open an issue or a pull request.

## PLEASE LEAVE A :star: TO SUPPORT THIS WORK
