import argparse
import cv2
from pydub import AudioSegment, silence
from scipy.fftpack import fft
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from tqdm import tqdm

def find_tone_segments(audio, target_freq=5000, threshold=0.1, chunk_size=100):
    """
    Trova segmenti audio con toni a una frequenza target.

    :param audio: L'audio creato con AudioSegment.from_file(input_video)
    :param target_freq: Frequenza target (in Hz). In Audacity tono da 10000hz su traccia stereo
    :param threshold: Valore soglia per rilevare la presenza della frequenza.
    :param chunk_size: Durata del chunk audio in millisecondi da analizzare.
    :return: Lista di tuple con gli intervalli temporali (start, end) in millisecondi.
    """
    duration_ms = len(audio)

    tone_segments = []
    current_start = None

    for start_ms in tqdm(range(0, duration_ms, chunk_size), desc="Analizzando audio"):
        chunk = audio[start_ms:start_ms + chunk_size]
        samples = np.array(chunk.get_array_of_samples())
        fft_result = np.abs(fft(samples))
        freqs = np.fft.fftfreq(len(fft_result), d=1 / chunk.frame_rate)

        # Trova l'indice della frequenza target
        idx = np.where((freqs >= target_freq - 5) & (freqs <= target_freq + 5))[0]

        # Controlla se la frequenza target supera la soglia
        if idx.size > 0 and np.max(fft_result[idx]) > threshold:
            if current_start is None:
                current_start = start_ms
        else:
            if current_start is not None:
                tone_segments.append((current_start, start_ms))
                current_start = None

    # Aggiungi l'ultimo segmento se ancora aperto
    if current_start is not None:
        tone_segments.append((current_start, duration_ms))

    return tone_segments

def find_silence(audio, silence_len=1000, silence_thresh=-80):
    """
    Trova i silenzi lunghi almeno silence_len millisecondi e con massimo silence_thres decibel
    :return: Lista di tuple con gli intervalli temporali (start, end) in millisecondi.
    """
    with tqdm(total=len(audio), desc="Analisi del silenzio", unit="ms") as pbar:
        sil = silence.detect_silence(
            audio,
            min_silence_len=silence_len,
            silence_thresh=silence_thresh,
            seek_step=10  # Controlla ogni 10ms per migliorare la velocità
        )
        pbar.update(len(audio))
    dead_time = [(0,sil[0][0])] if sil else []
    for start, stop in sil:
        dead_time.append((start+100, stop-100))

    return dead_time

def main(input_video, output_file, freq):
    # Estrai l'audio dal video
    video = VideoFileClip(input_video)
    audio =  AudioSegment.from_file(input_video)

    # Trova i segmenti con toni sinusoidali a 10000 Hz
    # In Audacity genera toni da 10000hz su traccia stereo, ma cerca 5000hz in python
    tone_segments = find_tone_segments(audio, target_freq=freq, threshold=1e6)
    # Trova i silenzi
    silence = find_silence(audio)
    # Unisce le due liste e le ordina, trasformando i millisecondi in secondi.
    tone_segments += silence
    merged = sorted(set(tone_segments))
    merged = [(start /1000, stop /1000 ) for start , stop in merged]
    print(merged)

    # Crea una lista dei segmenti che devono essere essere mantenuti.
    segmenti_attivi = []
    inizio = 0

    for start, stop in merged:
        if inizio < start:
            clip = video.subclip(inizio, start)

            clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
            segmenti_attivi.append(clip)
        inizio = stop
    video_finale = concatenate_videoclips(segmenti_attivi)
    video_finale.write_videofile(output_file, codec="h264_nvenc", audio_codec="aac")
    video_finale.close()
    video.reader.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rileva toni sinusoidali a 1000 Hz in un video.")
    parser.add_argument("input_video", default="", help="Percorso al file video di input.")
    parser.add_argument("output_file", help="Percorso al file di output per salvare gli intervalli.")
    parser.add_argument("freq", default=5000, type=int)
    args = parser.parse_args()

    main(args.input_video, args.output_file, args.freq)
