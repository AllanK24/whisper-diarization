import argparse
import logging
import os
import re

import torch
import torchaudio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from transcription_helpers import transcribe_batched_performance

mtypes = {"cpu": "int8", "cuda": "float16"}


def diarize_performance(
    model,
    alignment_model,
    alignment_tokenizer,
    punct_model,
    audio_file: str,
    language: str,
    batch_size: int,
    device: str,
    output_dir: str,
):
    
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_file}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = audio_file
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(audio_file))[0],
            "vocals.wav",
        )
        
    
    # Transcribe the audio file

    whisper_results, language, audio_waveform = transcribe_batched_performance(
        whisper_model=model,
        audio_file=vocal_target,
        language=language,
        batch_size=batch_size,
    )
    
    audio_waveform = (
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device)
    )
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )
    
    full_transcript = "".join(segment["text"] for segment in whisper_results)

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[language],
    )
    
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )
    
    
    spans = get_spans(tokens_starred, segments, blank_token)

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    
    # convert audio to mono for NeMo combatibility
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        audio_waveform.cpu().unsqueeze(0).float(),
        16000,
        channels_first=True,
    )
    
    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()
    
    # Reading timestamps <> Speaker Labels mapping


    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    with open(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.txt"), "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.srt"), "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
