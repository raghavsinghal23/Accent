import librosa
import soundfile as sf
import torch
import torch.nn as nn
import whisper
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

def extract_features(audio, sr=16000):
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(inputs.input_values)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# Dummy accent categories
ACCENT_CLASSES = ["American", "British", "Indian", "Australian", "Others"]

class AccentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AccentClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Initialize classifier (assuming 1024-dimensional input from Wav2Vec2)
accent_classifier = AccentClassifier(1024, len(ACCENT_CLASSES))


def predict_accent(audio_path):
    audio = load_audio(audio_path)
    print("Audio",audio)
    features = extract_features(audio)  # (1, 1024)
    print("features",features)
    logits = accent_classifier(features)
    print("logits",logits)
    predicted_index = torch.argmax(logits, dim=1).item()
    print("predicted_index",predicted_index)
    return ACCENT_CLASSES[predicted_index]




def whisper_transcribe(audio_path):
    whisper_model = whisper.load_model("medium")
    result = whisper_model.transcribe("audio.wav")

    # Print the detected text and optionally language
    print("Detected Language:", result["language"])
    print("Transcription:", result["text"])
    print("result", result)
    # result = model.transcribe(audio_path, language="en")
    return result['text']


if __name__ == "__main__":
    test_audio = "audio.wav"
    predicted = predict_accent(test_audio)
    print(f"Detected Accent: {predicted}")
            
    text = whisper_transcribe(test_audio)
     

