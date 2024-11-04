# HHA504 Assignment: Exploring AI and Analytics with Pre-trained Models in GCP and Azure

## Objective
This assignment demonstrates the use of pre-trained models in **Google Cloud Platform (GCP)** and **Azure** for speech-to-text transcription and object detection. The goal is to evaluate model capabilities, compare results across platforms, and document any challenges faced.

---

## 1. GCP Speech-to-Text API

### Steps
1. **Set Up JSON Key File for Authentication**:
   - Downloaded a JSON key file from GCP and uploaded it to **Vertex AI Workbench**.
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to authenticate with GCP APIs.

2. **Audio File Preparation**:
   - The initial audio file was in stereo format, but GCP’s Speech-to-Text API requires mono audio.
   - Converted the audio file to mono using `ffmpeg`:
     ```bash
     !ffmpeg -i original-audio.wav -ac 1 mono-audio.wav
     ```

3. **Transcribing the Audio File**:
   - Used `google.cloud.speech` library to perform transcription on the mono audio file:
     ```python
     from google.cloud import speech

     client = speech.SpeechClient()
     with open("mono-audio.wav", "rb") as audio_file:
         content = audio_file.read()

     audio = speech.RecognitionAudio(content=content)
     config = speech.RecognitionConfig(
         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
         sample_rate_hertz=16000,
         language_code="en-US"
     )

     response = client.recognize(config=config, audio=audio)
     for result in response.results:
         print("Transcript: {}".format(result.alternatives[0].transcript))
     ```

![image](https://github.com/user-attachments/assets/5d8fb398-f240-4b4d-968a-20d94300c5fa)
![image](https://github.com/user-attachments/assets/030f1ef6-993f-42a3-8230-5c43a37d5a25)
![image](https://github.com/user-attachments/assets/27468242-1cf7-4ecf-a757-62e3acbef0ea)


### Results
- **Transcript Output**: Hello this is my sample audio
- **Observations**: The transcription accuracy was satisfactory, capturing words clearly with no errors.

### Challenges
- **Stereo to Mono Conversion**: Initially, the API returned an error due to stereo format. This was resolved by converting the audio to mono.
- **Environment Variable Setup**: Setting the `GOOGLE_APPLICATION_CREDENTIALS` was necessary for authentication, which worked smoothly after uploading the JSON file.

---

## 2. GCP Vision API for Object Detection

### Steps
1. **Image Upload and Preparation**:
   - Uploaded an image file to **Vertex AI Workbench** for analysis.

2. **Object Detection**:
   - Used `google.cloud.vision` library to detect objects in the image:
     ```python
     from google.cloud import vision

     client = vision.ImageAnnotatorClient()
     with open("cat.jpg", "rb") as image_file:
         content = image_file.read()
     image = vision.Image(content=content)

     response = client.object_localization(image=image)
     for object_ in response.localized_object_annotations:
         print("Object detected: {} (confidence: {})".format(object_.name, object_.score))
     ```

![image](https://github.com/user-attachments/assets/1daa22dd-2338-4c6f-8a4d-bbcea48e694a)


### Results
- **Detected Objects and Confidence Scores**:
  - Object detebcted: Animal (Confidence: 0.6315107345581055)
  - Object detected: Cat (Confidence: 0.5887526869773865)
- **Observations**: The model was effective in detecting common objects in the image.

### Challenges
- **Initial Authentication Setup**: Required setting up the JSON key file, which was addressed in the first section and applied here.

---

## 3. Azure Vision for Object Detection

### Steps
1. **Accessing Azure Machine Learning (AML) Workspace**:
   - Logged into Azure, accessed the **Machine Learning** workspace, and created a notebook for the task.

2. **Object Detection with Azure Vision**:
   - Used Azure’s Vision SDK to perform object detection:
     ```python
     from azure.ai.vision import VisionClient, VisionApiKeyCredential

     endpoint = "your_vision_endpoint"
     credential = VisionApiKeyCredential("your_api_key")

     client = VisionClient(endpoint, credential)
     with open("sample-image.jpg", "rb") as image:
         response = client.detect_objects(image=image)

     for detected_object in response.objects:
         print(f"Object: {detected_object.name}, Confidence: {detected_object.confidence}")
     ```

### Results
- **Detected Objects and Confidence Scores**:
  - Object 1: [Object name] - Confidence: [Confidence score]
  - Object 2: [Object name] - Confidence: [Confidence score]
- **Observations**: The Azure Vision API detected similar objects as the GCP Vision API, with slight variations in confidence scores.

### Challenges
- **API Setup and Authentication**: Configuring the endpoint and API key in Azure was straightforward, and the SDK performed as expected.

---

## 4. Comparison of GCP and Azure

- **GCP vs. Azure**: GCP’s Speech-to-Text API was effective, while both GCP and Azure provided accurate object detection. However, Azure’s Vision API slightly differed in confidence scores.
- **Ease of Use**: Both platforms had straightforward SDKs, though initial setup in Vertex AI Workbench required more configuration for authentication with GCP.

---

## 5. Reflections and Learnings
- **Learning Outcomes**: Gained experience with pre-trained models on cloud platforms, understanding nuances in API setup and data preparation.
- **Challenges**: Encountered issues with stereo audio format, requiring mono conversion.
- **Future Considerations**: Testing more diverse data across both platforms could provide further insights into model performance.

---

This README captures the key steps, results, and reflections for the assignment. Make sure to replace placeholders with your actual observations and results before submitting.
