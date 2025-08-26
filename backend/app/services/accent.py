import os
import numpy as np
import torch
import torchaudio
import librosa
from speechbrain.pretrained import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import whisper
from datetime import datetime
from typing import Dict, Any
import tempfile
import cloudinary
import cloudinary.uploader
from app.config import settings

class ArabicAccentComparator:
    def __init__(self):
        self.speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_speaker_model"
        )
        self.whisper_model = None
        self.SAMPLE_RATE = 16000
        self.MFCC_N = 13

    def load_audio(self, audio_content: bytes) -> tuple:
        """Load audio from bytes"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_content)
            tmp_file.flush()
            
            waveform, sample_rate = torchaudio.load(tmp_file.name)
            
            if sample_rate != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.SAMPLE_RATE
                )
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            os.unlink(tmp_file.name)
            return waveform.numpy().squeeze(), self.SAMPLE_RATE

    def extract_speaker_embedding(self, audio_content: bytes) -> np.ndarray:
        """Extract speaker embedding"""
        waveform, _ = self.load_audio(audio_content)
        waveform = torch.tensor(waveform).unsqueeze(0)
        embedding = self.speaker_model.encode_batch(waveform)
        return embedding.squeeze(0).cpu().numpy()

    def compare_speaker_embeddings(self, audio_content1: bytes, audio_content2: bytes) -> float:
        """Compare speaker embeddings"""
        emb1 = self.extract_speaker_embedding(audio_content1)
        emb2 = self.extract_speaker_embedding(audio_content2)
        
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        return (similarity + 1) / 2

    def extract_acoustic_features(self, audio_content: bytes) -> Dict[str, Any]:
        """Extract acoustic features"""
        y, sr = self.load_audio(audio_content)
        
        features = {
            "rms_energy": librosa.feature.rms(y=y)[0],
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(y)[0],
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
            "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.MFCC_N),
            "pitch": librosa.pyin(y, fmin=80, fmax=400, sr=sr)[0]
        }
        return features

    def compare_acoustic_features(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """Compare acoustic features"""
        comparisons = {}
        
        for feature_name in features1:
            if feature_name == "mfcc":
                dist, _ = fastdtw(
                    features1[feature_name].T,
                    features2[feature_name].T,
                    dist=euclidean
                )
                comparisons[f"mfcc_dtw_distance"] = dist
            elif feature_name == "pitch":
                valid_pitch1 = features1[feature_name][~np.isnan(features1[feature_name])]
                valid_pitch2 = features2[feature_name][~np.isnan(features2[feature_name])]
                min_len = min(len(valid_pitch1), len(valid_pitch2))
                if min_len > 1:
                    comparisons["pitch_correlation"] = np.corrcoef(
                        valid_pitch1[:min_len], valid_pitch2[:min_len]
                    )[0,1]
                else:
                    comparisons["pitch_correlation"] = 0.0
            else:
                min_len = min(len(features1[feature_name]), len(features2[feature_name]))
                if min_len > 1:
                    comparisons[f"{feature_name}_correlation"] = np.corrcoef(
                        features1[feature_name][:min_len], features2[feature_name][:min_len]
                    )[0,1]
                else:
                    comparisons[f"{feature_name}_correlation"] = 0.0
        
        return comparisons

    def transcribe_audio(self, audio_content: bytes) -> str:
        """Transcribe Arabic audio"""
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("medium", device=device)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_content)
            tmp_file.flush()
            result = self.whisper_model.transcribe(tmp_file.name, language="ar")
            os.unlink(tmp_file.name)
            return result["text"]

    def compare_pronunciation(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare pronunciation"""
        return {
            "levenshtein_distance": self._levenshtein_distance(text1, text2),
            "length_difference": abs(len(text1) - len(text2))
        }

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def generate_comparison_visualization(self, audio_content1: bytes, audio_content2: bytes) -> str:
        """Generate and upload visualization to Cloudinary"""
        import matplotlib.pyplot as plt
        import librosa.display
        
        y1, sr1 = self.load_audio(audio_content1)
        y2, sr2 = self.load_audio(audio_content2)
        
        plt.figure(figsize=(15, 10))
        
        # Waveform comparison
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y1, sr=sr1, alpha=0.5, label="Reference")
        librosa.display.waveshow(y2, sr=sr2, alpha=0.5, label="Comparison")
        plt.title("Waveform Comparison")
        plt.legend()

        # Spectrogram comparison
        plt.subplot(3, 1, 2)
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
        librosa.display.specshow(D1, y_axis='log', x_axis='time', sr=sr1, alpha=0.5)
        librosa.display.specshow(D2, y_axis='log', x_axis='time', sr=sr2, alpha=0.5)
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram Comparison")

        # MFCC comparison
        plt.subplot(3, 1, 3)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=self.MFCC_N)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=self.MFCC_N)
        plt.plot(mfcc1.mean(axis=1), label="Reference")
        plt.plot(mfcc2.mean(axis=1), label="Comparison")
        plt.title("MFCC Comparison")
        plt.xlabel("MFCC Coefficient")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        
        # Save to temporary file and upload to Cloudinary
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            upload_result = cloudinary.uploader.upload(
                tmp_file.name,
                folder="accent_comparisons",
                transformation=[{"quality": "auto", "fetch_format": "auto"}]
            )
            os.unlink(tmp_file.name)
            
            return upload_result["secure_url"]

    def compare_accents(self, audio_content1: bytes, audio_content2: bytes) -> Dict[str, Any]:
        """Comprehensive accent comparison"""
        # Speaker similarity
        speaker_sim = self.compare_speaker_embeddings(audio_content1, audio_content2)
        
        # Acoustic features
        features1 = self.extract_acoustic_features(audio_content1)
        features2 = self.extract_acoustic_features(audio_content2)
        acoustic_comparison = self.compare_acoustic_features(features1, features2)
        
        # Transcription and pronunciation
        text1 = self.transcribe_audio(audio_content1)
        text2 = self.transcribe_audio(audio_content2)
        pronunciation_diff = self.compare_pronunciation(text1, text2)
        
        # Visualization
        visualization_url = self.generate_comparison_visualization(audio_content1, audio_content2)
        
        # Calculate overall score
        weights = {"speaker_similarity": 0.3, "acoustic_features": 0.5, "pronunciation": 0.2}
        
        acoustic_scores = []
        for name, value in acoustic_comparison.items():
            if "distance" in name:
                acoustic_scores.append(1 / (1 + value))
            else:
                acoustic_scores.append((value + 1) / 2)
        
        avg_acoustic = np.mean(acoustic_scores) if acoustic_scores else 0.0
        
        max_len = max(len(text1), len(text2)) or 1
        pronunciation_sim = 1 - (pronunciation_diff["levenshtein_distance"] / max_len)
        
        overall_score = (
            weights["speaker_similarity"] * speaker_sim +
            weights["acoustic_features"] * avg_acoustic +
            weights["pronunciation"] * pronunciation_sim
        )
        
        return {
            "speaker_similarity": float(speaker_sim),
            "acoustic_comparison": {k: float(v) for k, v in acoustic_comparison.items()},
            "pronunciation_differences": pronunciation_diff,
            "transcriptions": {"reference": text1, "comparison": text2},
            "visualization_url": visualization_url,
            "overall_score": float(overall_score)
        }

# Singleton instance
accent_comparator = ArabicAccentComparator()