# Web Service for SpeechCraft

![image of openapi server](server_screenshot.png)


# Starting the Web Service

In your cmd start the server with:
- `python -m speechcraft.server` 

Note: The first time you start the server, it will download the models. This can take a while.

# How it works:

The Webservice is built with [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI). 
In this regard, for each request it will create a task and return a job id.
You can then check the status of the job and retrieve the result.

We recommend using [fastSDK](https://github.com/SocAIty/fastSDK) and [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI) as illustrated for easy file transfers.

<img src="https://github.com/SocAIty/FastTaskAPI/blob/main/docs/fastsdk_to_fasttaskapi.png?raw=true" width="50%" />

Read the documentations of [fastSDK](https://github.com/SocAIty/fastSDK), [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI) and 
[media-toolkit](https://github.com/SocAIty/media-toolkit) to get the most out of the service and to familiarize yourself with the concepts.


# Configuration

You can configure some settings via environment variables:

| ENV Variable | Description                                                                                                                             |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| MODELS_DIR | Path to the folder where the models are stored. For example the inswapper and the GPEN models are downloaded and stored in this folder. |
| EMBEDDINGS_DIR | Path to the folder where the face embeddings are stored. Stored faces can be reused by the api.                                         |


# Deployment, Runpod, Docker, file uploads and more

For more settings and how to deploy the service check-out [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI).
For example it allows you to deploy the service with [Runpod](https://runpod.io) out of the box.


# Usage

The webservice provides enpdoints for the swap, add_face and enhance_face functionality.
You can send requests to the endpoints with any http client, e.g. requests, httpx, curl, etc.

Using [fastSDK](https://github.com/SocAIty/fastSDK) is the most convenient way to interact with the webservice.
You can find an implementation of an SDK generated for fastSDK in the [socaity SDK](https://github.com/SocAIty/socaity/tree/main/socaity/api/image/img2img/face2face) documentation.


### With fastSDK

FastSDK is a python module that provides a convenient way to interact with the webservice.
You can find an implementation of an SDK generated for fastSDK in the [socaity SDK](https://github.com/SocAIty/socaity/tree/main/socaity/api/image/img2img/face2face) documentation.


### With plain web requests
### Send requests

```python
from media_toolkit import AudioFile

# text2speech
response = httpx.post("http://localhost:8009/text2voice", params={ "text" : "please contribute", "voice": "en_speaker_3"})

# voice cloning
audio = AudioFile().from_file("myfile.wav")
request = httpx.post(
   "http://localhost:8009/voice2embedding", params={ "voice_name" : "hermine"}, 
    files={"audio": audio.to_httpx_send_able_tuple()}
)

# voice2voice  
response = httpx.post(
    "http://localhost:8009/voice2voice", 
    params={ "voice_name" : "hermine"}, 
    files={"audio": audio.to_httpx_send_able_tuple()}
)
```

This example shows how to do it with plain python requests.

```python
import requests

# text2speech synthesis
response = requests.post("http://localhost:8009/text2voice", params={ "text" : "please contribute", "voice": "en_speaker_3"})

# Speaker embedding creation
with open("myfile.wav", "rb") as f:
    audio = f.read()
    
response = requests.post(
    "http://localhost:8009/voice2embedding", 
    params={ "voice_name" : "my_new_speaker"}, 
    files={"audio_file": audio}
)

# voice2voice
response = requests.post(
    "http://localhost:8009/voice2voice", 
    params={ "voice_name" : "my_new_speaker"}, 
    files={"audio_file": audio}
)
```
### Parse the results

The response is a json that includes the job id and meta information.
By sending then a request to the job endpoint you can check the status and progress of the job.
If the job is finished, you will get the result, including the swapped image.
```python
import requests
from media_toolkit import AudioFile

# check status of job
response = requests.get(f"http://localhost:8009/api/job/{job.json()['job_id']}")
# convert result to image file
audio = AudioFile().from_bytes(response.json()['result']))
```


# Please leave a :star: to support us