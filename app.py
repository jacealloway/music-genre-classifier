# Import packages
import gradio as gr 
import numpy as np 
import librosa
import joblib 
import pandas as pd 
from keras.models import load_model, Model
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set the base directory (for local computing - Huggingface should ignore this since the package works out of BASE anyway)
base = os.path.dirname(os.path.abspath(__file__))
artifacts  = os.path.join(base, 'artifacts')

# Load the exported models from the audio_classifier.ipynb training notebook
model_mel = load_model(os.path.join(artifacts, 'model_mel.keras'))
model_tempogram = load_model(os.path.join(artifacts, 'model_tempogram.keras'))
model_chroma = load_model(os.path.join(artifacts, 'model_chroma.keras'))
xgbmodel = joblib.load(os.path.join(artifacts, 'xgbmodel.pkl'))
scaler = joblib.load(os.path.join(artifacts, 'scaler.pkl'))
le = joblib.load(os.path.join(artifacts, 'label_encoder.pkl'))

# Define the global parameters: sampling rate, spectral hop, number of Mel-Frequency Spectrum Values (for CNN input), and clip trim duration
SR, HOP, N_MEL, DUR = 22_050, 2_048, 128, 3


def emb(model: Model) -> Model:
    """
    Embedding function to extract the embedded final layers of each CNN for spectral images
    """
    return Model(model.inputs, model.get_layer('embedding').output)

def aggregate(array: np.array) -> np.array:
    """
    Aggregation function for computing mean, standard deviation, maximum, minimum of a statistical vector input
    """
    return np.concatenate([array.mean(axis = 1), array.std(axis = 1), array.max(axis = 1), array.min(axis = 1)])

class Audiofile:
    """
    A class containing an audio file and feature attributes

    ...

    Methods
    ----------

    extract_stat_features() -> tuple[np.array]
        Computes statistical embeddings of audio file features:
            Mel-Frequency Cepstral Coefficients (MFCCs)
            Mel-Frequency Cepstral Coefficient Deltas (MFCC Deltas)
            Tempo 
            Frequency Rolloff at 1% (Minimum)
            Frequency Rolloff at 99% (Maximum)
            Root Mean Square
            Zero-Crossing Rate
            Tone Network (Tonnetz)
            Spectral Contrast 
            Centre / Centroid Frequency

    @staticmethod
    
    fix_length(y: np.array, target: int) -> np.array
        Trims or 0-pads an array.

    fix_frames(spec: np.array) -> np.array:
        Trims or 0-pads an image / 2-dimensional array).

    normalize_spectogram(spec: np.array) -> np.array
        Normalizes and scales an image / 2-dimensional array by transforming to the minimum value and mapping to [0, 1].

    extract_spec_features() -> tuple[np.array]:
        Computes the spectral renderings of audio file features:
            Mel-Frequency Spectogram, normalized to dB log-10 scale
            Tempogram 
            Chromagram 
    """
    def __init__(self, series: np.array, sr: int, duration: int, spec_hop: int) -> None:
        self.y = series 
        self.sr = sr 
        self.duration = duration
        self.spec_hop = spec_hop

    def __str__(self):
        return 'Audiofile object'


    def extract_stat_features(self) -> tuple[np.array]:
        """
        Computes the Librosa features of an audio file.
        """
        mfcc = librosa.feature.mfcc(y = self.y, sr = self.sr, n_mfcc = 20)
        mfcc_deltas = librosa.feature.delta(mfcc)
        tempo = librosa.feature.tempo(y = self.y, sr = self.sr)
        rolloff_001 = librosa.feature.spectral_rolloff(y = self.y, sr = self.sr, roll_percent = 0.01)
        rolloff_099 = librosa.feature.spectral_rolloff(y = self.y, sr = self.sr, roll_percent = 0.99)
        rms = librosa.feature.rms(y = self.y)
        zcr = librosa.feature.zero_crossing_rate(y = self.y)
        tonnetz = librosa.feature.tonnetz(y = self.y, sr = self.sr)
        contrast_bands = librosa.feature.spectral_contrast(y = self.y, sr = self.sr)
        centroid_freq = librosa.feature.spectral_centroid(y = self.y, sr = self.sr)

        return mfcc, mfcc_deltas, tempo, rolloff_001, rolloff_099, rms, zcr, tonnetz, contrast_bands, centroid_freq


    # Calling outside the class 
    @staticmethod
    def fix_length(y: np.array, target: int = SR * DUR) -> np.array:
        """
        Fix the length of an array.
        """
        if len(y) > target:
            return y[ : target]
        return np.pad(y, (0, target - len(y)))

    def fix_frames(self, spec: np.array) -> np.array:
        """
        Fix the length and width of an image / 2-dimensional array.
        """
        target = 1 + (self.sr * self.duration) // self.spec_hop

        time_dim = spec.shape[1]
        if time_dim > target:
            return spec[:, :target]
        return np.pad(spec, ((0,0), (0, target - time_dim)))
    

    def normalize_spectogram(self, spec: np.array) -> np.array:
        mn, mx = spec.min(), spec.max()
        if mx - mn < 1e-8:
            return np.zeros_like(spec)
        return (spec - mn) / (mx - mn)


    def extract_spec_features(self) -> tuple[np.array]:
        tempogram = librosa.feature.tempogram(y = self.y, sr = self.sr)
        mel_spec = librosa.feature.melspectrogram(y = self.y, sr = self.sr)
        mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
        chromagram = librosa.feature.chroma_cens(y = self.y, sr = self.sr)
        
        return (self.normalize_spectogram(self.fix_frames(tempogram)), 
                self.normalize_spectogram(self.fix_frames(mel_spec)), 
                self.normalize_spectogram(self.fix_frames(chromagram)))


mel_ext, tempo_ext, chroma_ext = emb(model_mel), emb(model_tempogram), emb(model_chroma)



def classify(audio_path: str) -> dict:
    """
    Model algorithm function for classifying music genres
    """

    # Initialize blank audio dictionary for storing Audiofile objects per-analysis (dependent on the number of segmented cuts of the time series)
    audio_dict = {
        'audio_object' : []
    }

    # Fail-safe: try/except incase there is an issue with the audio file
    try:
        y, _ = librosa.load(audio_path, sr = SR)
        # Define segmented length
        seg_length = SR * DUR

        # If the audio is too short, pad it with zeros
        if len(y) < seg_length:
            y = np.pad(y, (0, seg_length - len(y)))

        # Extract each segmented clip and store as Audiofile objects
        for i, start in enumerate(range(0, len(y)-seg_length+1, seg_length)):
                    seg = y[start:start+seg_length]
                    audio_dict['audio_object'].append(Audiofile(series=seg, sr=SR, duration=DUR, spec_hop=HOP))
    
    except Exception as e:
        print(f'Error opening file {audio_path}: {e}.')

    # Error-handling: if user inputs a 0-length file, raise a ValueError 
    if len(y) == 0:
        raise ValueError('Input waveform has 0 length.')
    
    # Transform dictionary to Pandas dataframe for easy access
    AUDIO = pd.DataFrame(audio_dict)

    # Extract the spectral features
    specs = [a.extract_spec_features() for a in AUDIO['audio_object'].values]
    tempo_arr_pre  = np.stack([s[0] for s in specs])
    mel_arr_pre    = np.stack([s[1] for s in specs])
    chroma_arr_pre = np.stack([s[2] for s in specs])

    # Pad the arrays with an empty axis to match the CNN input for Conv2D
    tempo_arr = tempo_arr_pre[..., np.newaxis]
    mel_arr = mel_arr_pre[..., np.newaxis]
    chroma_arr = chroma_arr_pre[..., np.newaxis]

    # Use Tensorflow to resize each image for CNN input, store as Numpy arrays
    tempo_small = tf.image.resize(tempo_arr, [128, 128]).numpy()
    mel_small   = tf.image.resize(mel_arr,   [128, 128]).numpy()
    chroma_small= tf.image.resize(chroma_arr,[12, 128]).numpy()  # keep 12 pitch rows

    # Convert to float32 for memory optimization
    tempo_small = tempo_small.astype('float32')
    mel_small   = mel_small.astype('float32')
    chroma_small= chroma_small.astype('float32')

    # Extract the embeddings of each input as features
    tempo_emb  = tempo_ext.predict(tempo_small, batch_size=16)  
    mel_emb    = mel_ext.predict(mel_small, batch_size=16)   
    chroma_emb = chroma_ext.predict(chroma_small, batch_size=16) 

    # Concat all arrays
    X_cnn = np.concatenate([tempo_emb, mel_emb, chroma_emb], axis=1) 


    # Initialize blank statistical list 
    stat_vectors = []

    # Iterate through each clip 
    for audio in AUDIO['audio_object'].values:
        # Get statistical vectors from audio
        mfcc, mfcc_deltas, tempo, rolloff_001, rolloff_099, rms, zcr, tonnetz, contrast_bands, centroid_freq = audio.extract_stat_features()

        # Use aggregate() to compute the attributes of each array, store them as features
        features = np.concatenate([
            aggregate(mfcc),
            aggregate(mfcc_deltas),
            aggregate(rolloff_001),
            aggregate(rolloff_099),
            aggregate(rms),
            aggregate(zcr),
            aggregate(tonnetz),
            aggregate(contrast_bands),
            aggregate(centroid_freq),
            np.atleast_1d(tempo)
        ])

        # Add to stat vector feature list
        stat_vectors.append(features)

    # Convert list to np.array
    X_stat = np.array(stat_vectors)
    # Scale the vectors based on the scaler from training 
    X_stat_scaled = scaler.transform(X_stat)
    # Concat all features 
    X_total = np.concatenate([X_cnn, X_stat_scaled], axis = 1)
    # Predict the probability of the genre: return a series of floats from XGBoost output
    proba = xgbmodel.predict_proba(X_total).mean(axis=0)

    # Return the labeled classes from the Label Encoder
    return {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}



# Initialize Gradio demo for interface host 
demo = gr.Interface(
    fn = classify,  # Classify model
    # For audio input - filepath will be via upload
    inputs = gr.Audio(type = 'filepath', label = 'Upload a music clip.'),   
    # Return the top 3 probabilities with labels to show the predicted genre
    outputs = gr.Label(num_top_classes = 3, label = 'Predicted genre'), 

    # Add description and title
    title = 'Music Genre Classifier',
    description = 'CNN deep features + XGBoost. Trained on GTZAN.'
)

if __name__ == '__main__':
    # Run the model demo
    demo.launch(
        server_name = '0.0.0.0',
        server_port = int(os.environ.get('PORT', 7860))
    )