"""
MNE-Python integration for loading standard EEG file formats.

Supports .edf, .bdf, .fif, .set, and .vhdr formats with automatic
preprocessing and feature extraction pipelines.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

from neuroformer.utils.logging import get_logger
from neuroformer.utils.exceptions import DataValidationError, PreprocessingError

logger = get_logger(__name__)


STANDARD_10_20 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

SUPPORTED_FORMATS = ['.edf', '.bdf', '.fif', '.set', '.vhdr', '.cnt', '.gdf']


class MNELoader:
    """
    Load EEG data from standard file formats using MNE-Python.
    
    Provides a unified interface for loading, preprocessing, and
    extracting features from various EEG file formats.
    """
    
    def __init__(
        self,
        sampling_rate: int = 256,
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        notch_freq: Optional[float] = 60.0,
        reference: str = 'average',
        channels: Optional[List[str]] = None,
        reject_bad: bool = True,
        ica_components: int = 15
    ):
        """
        Args:
            sampling_rate: Target sampling rate (Hz)
            l_freq: Low cutoff for bandpass filter
            h_freq: High cutoff for bandpass filter
            notch_freq: Notch filter frequency (None to skip)
            reference: Re-reference type ('average', 'REST', channel name)
            channels: Channels to select (None = all standard 10-20)
            reject_bad: Whether to reject bad channels/segments
            ica_components: Number of ICA components for artifact removal
        """
        self.sampling_rate = sampling_rate
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.reference = reference
        self.channels = channels or STANDARD_10_20
        self.reject_bad = reject_bad
        self.ica_components = ica_components
    
    def load_raw(self, filepath: str) -> 'mne.io.Raw':
        """
        Load raw EEG data from file.
        
        Args:
            filepath: Path to EEG file
            
        Returns:
            MNE Raw object
            
        Raises:
            DataValidationError: If file format is unsupported
        """
        import mne
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise DataValidationError(f"File not found: {filepath}")
        
        ext = filepath.suffix.lower()
        
        if ext not in SUPPORTED_FORMATS:
            raise DataValidationError(
                f"Unsupported file format: {ext}",
                expected=SUPPORTED_FORMATS,
                got=ext
            )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if ext in ['.edf', '.bdf']:
                raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
            elif ext == '.fif':
                raw = mne.io.read_raw_fif(str(filepath), preload=True, verbose=False)
            elif ext == '.set':
                raw = mne.io.read_raw_eeglab(str(filepath), preload=True, verbose=False)
            elif ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(str(filepath), preload=True, verbose=False)
            elif ext == '.cnt':
                raw = mne.io.read_raw_cnt(str(filepath), preload=True, verbose=False)
            elif ext == '.gdf':
                raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
        
        logger.info(f"Loaded {filepath.name}: {raw.info['nchan']} channels, "
                     f"{raw.times[-1]:.1f}s, {raw.info['sfreq']} Hz")
        
        return raw
    
    def preprocess(self, raw: 'mne.io.Raw') -> 'mne.io.Raw':
        """
        Apply full preprocessing pipeline.
        
        Steps:
        1. Pick EEG channels
        2. Channel selection
        3. Re-reference
        4. Resample
        5. Filter (bandpass + notch)
        6. ICA artifact removal (optional)
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Preprocessed Raw object
        """
        import mne
        
        raw = raw.copy()
        
        # Pick EEG channels only
        raw.pick_types(eeg=True)
        logger.info(f"After EEG pick: {raw.info['nchan']} channels")
        
        # Select standard channels if available
        available = [ch for ch in self.channels if ch in raw.ch_names]
        if len(available) < len(self.channels):
            missing = set(self.channels) - set(available)
            logger.warning(f"Missing channels: {missing}")
        
        if available:
            raw.pick_channels(available, ordered=True)
        
        # Re-reference
        if self.reference == 'average':
            raw.set_eeg_reference('average', projection=True, verbose=False)
            raw.apply_proj()
        elif self.reference != 'none':
            raw.set_eeg_reference([self.reference], verbose=False)
        
        # Resample
        if raw.info['sfreq'] != self.sampling_rate:
            raw.resample(self.sampling_rate, verbose=False)
            logger.info(f"Resampled to {self.sampling_rate} Hz")
        
        # Filter
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method='fir',
            fir_design='firwin',
            verbose=False
        )
        
        if self.notch_freq:
            raw.notch_filter(
                freqs=[self.notch_freq, self.notch_freq * 2],
                verbose=False
            )
        
        # ICA artifact removal
        if self.reject_bad and self.ica_components > 0:
            try:
                ica = mne.preprocessing.ICA(
                    n_components=min(self.ica_components, raw.info['nchan'] - 1),
                    random_state=42,
                    max_iter='auto'
                )
                ica.fit(raw, verbose=False)
                
                # Auto-detect EOG artifacts
                eog_indices = []
                try:
                    eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
                except Exception:
                    pass
                
                if eog_indices:
                    ica.exclude = eog_indices[:2]  # Exclude at most 2 components
                    raw = ica.apply(raw, verbose=False)
                    logger.info(f"Removed {len(ica.exclude)} ICA artifact components")
            except Exception as e:
                logger.warning(f"ICA failed, skipping: {e}")
        
        logger.info(f"Preprocessing complete: {raw.info['nchan']} channels, "
                     f"{raw.info['sfreq']} Hz")
        
        return raw
    
    def extract_epochs(
        self,
        raw: 'mne.io.Raw',
        epoch_duration: float = 4.0,
        overlap: float = 0.5
    ) -> np.ndarray:
        """
        Segment continuous data into fixed-length epochs.
        
        Args:
            raw: Preprocessed Raw object
            epoch_duration: Epoch length in seconds
            overlap: Overlap fraction (0 to 1)
            
        Returns:
            Epochs array (n_epochs, n_channels, n_samples)
        """
        import mne
        
        data = raw.get_data()  # (channels, samples)
        sfreq = raw.info['sfreq']
        
        epoch_samples = int(epoch_duration * sfreq)
        step_samples = int(epoch_samples * (1 - overlap))
        
        epochs = []
        start = 0
        
        while start + epoch_samples <= data.shape[1]:
            epoch = data[:, start:start + epoch_samples]
            epochs.append(epoch)
            start += step_samples
        
        epochs = np.array(epochs)
        logger.info(f"Extracted {len(epochs)} epochs of {epoch_duration}s")
        
        return epochs
    
    def extract_features(
        self,
        epochs: np.ndarray,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from epochs.
        
        Args:
            epochs: Epoch data (n_epochs, n_channels, n_samples)
            bands: Frequency band definitions
            
        Returns:
            Dictionary with 'band_powers', 'coherence', 'asymmetry'
        """
        from scipy.signal import welch
        from scipy.signal import coherence as sig_coherence
        
        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
        
        n_epochs, n_channels, n_samples = epochs.shape
        n_bands = len(bands)
        
        # Band powers
        all_band_powers = np.zeros((n_epochs, n_bands, n_channels))
        
        for ep_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                freqs, psd = welch(
                    epochs[ep_idx, ch_idx],
                    fs=self.sampling_rate,
                    nperseg=min(256, n_samples)
                )
                
                for b_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                    mask = (freqs >= fmin) & (freqs <= fmax)
                    all_band_powers[ep_idx, b_idx, ch_idx] = np.mean(psd[mask])
        
        # Coherence matrices (per epoch, per band)
        all_coherence = np.zeros((n_epochs, n_bands, n_channels, n_channels))
        
        for ep_idx in range(n_epochs):
            for i in range(n_channels):
                for j in range(i, n_channels):
                    f, coh = sig_coherence(
                        epochs[ep_idx, i],
                        epochs[ep_idx, j],
                        fs=self.sampling_rate,
                        nperseg=min(256, n_samples)
                    )
                    
                    for b_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                        mask = (f >= fmin) & (f <= fmax)
                        avg_coh = np.mean(coh[mask]) if mask.sum() > 0 else 0
                        all_coherence[ep_idx, b_idx, i, j] = avg_coh
                        all_coherence[ep_idx, b_idx, j, i] = avg_coh
        
        logger.info(f"Features extracted: band_powers {all_band_powers.shape}, "
                     f"coherence {all_coherence.shape}")
        
        return {
            'band_powers': all_band_powers,
            'coherence': all_coherence,
            'band_names': list(bands.keys()),
            'channel_names': list(range(n_channels))
        }
    
    def load_and_process(
        self,
        filepath: str,
        epoch_duration: float = 4.0,
        overlap: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Full pipeline: load → preprocess → epoch → extract features.
        
        Args:
            filepath: Path to EEG file
            epoch_duration: Epoch length in seconds
            overlap: Overlap fraction
            
        Returns:
            Feature dictionary
        """
        raw = self.load_raw(filepath)
        raw = self.preprocess(raw)
        epochs = self.extract_epochs(raw, epoch_duration, overlap)
        features = self.extract_features(epochs)
        
        return features


class BatchLoader:
    """
    Load and process multiple EEG files in batch.
    """
    
    def __init__(self, loader: Optional[MNELoader] = None):
        self.loader = loader or MNELoader()
    
    def load_directory(
        self,
        directory: str,
        labels: Optional[Dict[str, int]] = None,
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load all EEG files from a directory.
        
        Args:
            directory: Path to directory
            labels: Optional mapping of filename patterns to labels
            extensions: File extensions to include
            max_files: Maximum number of files to load
            
        Returns:
            Combined feature dictionary with labels
        """
        directory = Path(directory)
        extensions = extensions or SUPPORTED_FORMATS
        
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        
        files = sorted(files)
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Found {len(files)} EEG files in {directory}")
        
        all_powers = []
        all_coherence = []
        all_labels = []
        all_subject_ids = []
        
        for idx, filepath in enumerate(files):
            try:
                features = self.loader.load_and_process(str(filepath))
                n_epochs = features['band_powers'].shape[0]
                
                all_powers.append(features['band_powers'])
                all_coherence.append(features['coherence'])
                
                # Assign labels
                if labels:
                    label = self._match_label(filepath.name, labels)
                    all_labels.extend([label] * n_epochs)
                
                # Subject ID from filename
                all_subject_ids.extend([idx] * n_epochs)
                
                logger.info(f"[{idx+1}/{len(files)}] Loaded {filepath.name} ({n_epochs} epochs)")
                
            except Exception as e:
                logger.warning(f"Failed to load {filepath.name}: {e}")
                continue
        
        if not all_powers:
            raise DataValidationError("No files successfully loaded")
        
        result = {
            'band_powers': np.concatenate(all_powers, axis=0),
            'coherence': np.concatenate(all_coherence, axis=0),
            'subject_ids': np.array(all_subject_ids)
        }
        
        if all_labels:
            result['labels'] = np.array(all_labels)
        
        logger.info(f"Total: {result['band_powers'].shape[0]} epochs from {len(files)} files")
        
        return result
    
    def _match_label(self, filename: str, labels: Dict[str, int]) -> int:
        """Match filename to label using pattern matching."""
        filename_lower = filename.lower()
        for pattern, label in labels.items():
            if pattern.lower() in filename_lower:
                return label
        return -1  # Unknown
