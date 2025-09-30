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
import cloudinary
import cloudinary.uploader
from app.config import settings
import io
import soundfile as sf
from pydub import AudioSegment
import tempfile

class ArabicAccentComparator:
    def __init__(self):
        self.speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_speaker_model"
        )
        self.whisper_model = None
        self.SAMPLE_RATE = 16000
        self.MFCC_N = 13

    def load_audio_from_bytes(self, audio_content: bytes) -> tuple:
        """Load audio from bytes without temporary files"""
        try:
            # Try to load with soundfile first
            with io.BytesIO(audio_content) as audio_buffer:
                try:
                    y, sr = sf.read(audio_buffer)
                    # Convert to mono if stereo
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                    # Resample if needed
                    if sr != self.SAMPLE_RATE:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.SAMPLE_RATE)
                    return y, self.SAMPLE_RATE
                except:
                    # Fallback to torchaudio for other formats
                    audio_buffer.seek(0)
                    waveform, sr = torchaudio.load(audio_buffer)
                    
                    # Convert to numpy
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.numpy()
                    
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = np.mean(waveform, axis=0)
                    else:
                        waveform = waveform.squeeze()
                    
                    # Resample if needed
                    if sr != self.SAMPLE_RATE:
                        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.SAMPLE_RATE)
                    
                    return waveform, self.SAMPLE_RATE
                    
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")

    def extract_speaker_embedding(self, audio_content: bytes) -> np.ndarray:
        """Extract speaker embedding"""
        y, sr = self.load_audio_from_bytes(audio_content)
        
        # Convert to tensor format expected by speechbrain
        waveform = torch.tensor(y).unsqueeze(0).float()
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
        y, sr = self.load_audio_from_bytes(audio_content)
        
        features = {
            "rms_energy": librosa.feature.rms(y=y)[0],
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(y)[0],
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
            "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.MFCC_N),
            "pitch": librosa.pyin(y, fmin=80, fmax=400, sr=sr)[0],
            "ampitude":y
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
        """Transcribe Arabic audio without temporary files"""
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("small", device=device)
        
        # Convert audio to the format whisper expects (16kHz mono WAV)
        y, sr = self.load_audio_from_bytes(audio_content)
        
        # Whisper expects float32 audio in the range [-1, 1]
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        # Transcribe directly from numpy array
        result = self.whisper_model.transcribe(y, language="ar")
        return result["text"]


    def transcribe(self, audio_content: bytes) -> dict:
        """Transcribe Arabic audio with timestamps"""
        if self.whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model("small", device=device)
        
        # Convert audio to 16kHz mono
        y, sr = self.load_audio_from_bytes(audio_content)
    
        # Whisper expects float32 audio [-1, 1]
        if y.dtype != np.float32:
            y = y.astype(np.float32)
    
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
        
        # Ask Whisper for timestamps too
        result = self.whisper_model.transcribe(
            y,
            language="ar",
            verbose=False,
            word_timestamps=False  # set to True if you want per-word timings (only in some forks)
        )
    
        # Extract text + segment-level timings
        transcription = {
            "text": result["text"],
            "segments": [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip()
                }
                for seg in result["segments"]
            ]
        }
    
        return transcription

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
    
    def generate_comparison_visualization(self, audio_content1: bytes, audio_content2: bytes, transcription1: str = "", transcription2: str = "") -> str:
        """Generate and upload visualization to Cloudinary"""
        import matplotlib.pyplot as plt
        import librosa.display
        import numpy as np
        import io
        import cloudinary
        
        # Try to import Arabic text processing libraries
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
        except ImportError:
            # Fallback if libraries are not available
            arabic_reshaper = None
            get_display = None
    
        def format_arabic_text(text):
            """Format Arabic text for proper display"""
            if not text:
                return text
            if arabic_reshaper and get_display:
                # Reshape and apply bidirectional algorithm for Arabic
                reshaped_text = arabic_reshaper.reshape(text)
                return get_display(reshaped_text)
            return text
    
        # Configure matplotlib
        plt.rcParams['axes.unicode_minus'] = False
        try:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        except:
            pass
        
        y1, sr1 = self.load_audio_from_bytes(audio_content1)
        y2, sr2 = self.load_audio_from_bytes(audio_content2)
    
        plt.figure(figsize=(15, 14))
    
        # Sample exactly every 200 values
        n = 300
    
        # Create indices for sampling
        indices1 = np.arange(0, len(y1), n)
        indices2 = np.arange(0, len(y2), n)
    
        y1_sampled = y1[indices1]
        y2_sampled = y2[indices2]
    
        # Create time arrays for the sampled data
        time1 = indices1 / sr1
        time2 = indices2 / sr2
    
        # Format Arabic transcriptions
        formatted_transcription1 = format_arabic_text(transcription1)
        formatted_transcription2 = format_arabic_text(transcription2)
    
        # Waveform - Reference as line chart
        plt.subplot(4, 1, 1)
        plt.plot(time1, y1_sampled, alpha=0.8, color='blue', linewidth=1.2)
        
        if formatted_transcription1:
            plt.text(0.98, 1.05, "Reference Audio - Waveform", 
                    transform=plt.gca().transAxes, ha='right', fontsize=12, fontweight='bold')
            plt.text(0.98, 0.95, formatted_transcription1, 
                    transform=plt.gca().transAxes, ha='right', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            plt.title("Reference Audio - Waveform", loc='right')
            
        plt.ylabel("Amplitude")
        plt.xlabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
    
        # Waveform - Comparison as line chart
        plt.subplot(4, 1, 2)
        plt.plot(time2, y2_sampled, alpha=0.8, color='red', linewidth=1.2)
        
        if formatted_transcription2:
            plt.text(0.98, 1.05, "Comparison Audio - Waveform", 
                    transform=plt.gca().transAxes, ha='right', fontsize=12, fontweight='bold')
            plt.text(0.98, 0.95, formatted_transcription2, 
                    transform=plt.gca().transAxes, ha='right', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        else:
            plt.title("Comparison Audio - Waveform", loc='right')
            
        plt.ylabel("Amplitude")
        plt.xlabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
    
        # Rest of the code remains the same...
        plt.subplot(4, 1, 3)
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
        librosa.display.specshow(D1, y_axis='log', x_axis='time', sr=sr1, alpha=0.5, label="Reference")
        librosa.display.specshow(D2, y_axis='log', x_axis='time', sr=sr2, alpha=0.5, label="Comparison")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram Comparison")
    
        plt.subplot(4, 1, 4)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=self.MFCC_N)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=self.MFCC_N)
        plt.plot(mfcc1.mean(axis=1), label="Reference")
        plt.plot(mfcc2.mean(axis=1), label="Comparison")
        plt.title("MFCC Comparison")
        plt.xlabel("MFCC Coefficient")
        plt.ylabel("Value")
        plt.legend()
    
        plt.tight_layout()
    
        with io.BytesIO() as buffer:
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buffer.seek(0)
    
            upload_result = cloudinary.uploader.upload(
                buffer,
                folder="accent_comparisons",
                transformation=[{"quality": "auto", "fetch_format": "auto"}]
            )
    
            return upload_result["secure_url"]
    def compare_accents(self, audio_content1: bytes, audio_content2: bytes) -> Dict[str, Any]:
        """Comprehensive accent comparison"""
        try:
            # Speaker similarity
            speaker_sim = self.compare_speaker_embeddings(audio_content1, audio_content2)

            # Acoustic features
            features1 = self.extract_acoustic_features(audio_content1)
            features2 = self.extract_acoustic_features(audio_content2)
            acoustic_comparison = self.compare_acoustic_features(features1, features2)

            # Transcription and pronunciation (only once)
            text1 = self.transcribe_audio(audio_content1)
            text2 = self.transcribe_audio(audio_content2)
            pronunciation_diff = self.compare_pronunciation(text1, text2)

            # Visualization - pass transcriptions to avoid double transcription
            visualization_url = self.generate_comparison_visualization(
                audio_content1, 
                audio_content2, 
                text1, 
                text2
            )

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

        except Exception as e:
            raise Exception(f"Error in accent comparison: {str(e)}")
    # Singleton instance
accent_comparator = ArabicAccentComparator()