from time import time

import torch
import torchaudio

from ocotillo.model_loader import load_model
from ocotillo.utils import load_audio


class Transcriber:
    def __init__(self, phonetic=False, cuda_device=-1, use_torchscript=False):
        if cuda_device >= 0:
            self.device = f'cuda:{cuda_device}'
        else:
            self.device = 'cpu'
        self.model, self.processor = load_model(self.device, phonetic=phonetic, use_torchscript=use_torchscript)
        self.model.eval()

    def transcribe_batch(self, wave_b1t, sample_rate):
        wave_b1t = self._prepare_wave(wave_b1t, sample_rate)
        with torch.no_grad():
            logits_btc = self.model(wave_b1t.squeeze(1))[0]
            tokens = torch.argmax(logits_btc, dim=-1)
        return self.processor.batch_decode(tokens), logits_btc

    def _prepare_wave(self, wave_b1t, sample_rate):
        assert len(wave_b1t.shape) == 3, wave_b1t.shape
        assert wave_b1t.shape[1] == 1, wave_b1t.shape
        if sample_rate != 16000:
            wave_b1t = torchaudio.functional.resample(wave_b1t, sample_rate, 16000)
        # Normalize the wave.
        wave_b1t = (wave_b1t - wave_b1t.mean(2, keepdim=True)) / torch.sqrt(wave_b1t.var(2, keepdim=True) + 1e-7)
        wave_b1t = wave_b1t.to(self.device)
        return wave_b1t

    def transcribe_batch_with_time_chunking(self, wave_b1t, sample_rate, chunk_secs=22, overlap_secs=2):
        assert chunk_secs > overlap_secs, f"chunk_secs must be longer than 2*overlap_secs: {chunk_secs} > 2*{overlap_secs}"
        if chunk_secs >= (wave_b1t.shape[2] / sample_rate - 0.1):
            return self.transcribe_batch(wave_b1t, sample_rate=sample_rate)
        assert isinstance(chunk_secs, int)
        assert isinstance(overlap_secs, int)
        wav2vec_logit_unit = 320
        chunk_sz = wav2vec_logit_unit * chunk_secs * 50
        logits_overlap_sz = 50 * overlap_secs
        logits_sz = 50 * chunk_secs
        overlap_sz = wav2vec_logit_unit * logits_overlap_sz

        original_batch_size = wave_b1t.shape[0]
        wave_b1t = self._prepare_wave(wave_b1t, sample_rate)
        chunked_clips = []
        start = 0
        while 1:
            end = start + chunk_sz
            chunked_clips.append(wave_b1t[:, :, start:end])
            if end >= wave_b1t.shape[-1]:
                break
            start += chunk_sz - overlap_sz

        # Pad the last element to chunk_sz
        chunked_clips[-1] = torch.nn.functional.pad(chunked_clips[-1],
                                                    (0, chunk_sz - chunked_clips[-1].shape[-1]))
        chunked_clips = torch.cat(chunked_clips, dim=0)
        chunked_clips = torch.split(chunked_clips, original_batch_size, dim=0)

        logit_chunks = []
        last_logits = None
        for wave_b1t_chunk in chunked_clips:
            with torch.no_grad():
                logits_chunk_btc = self.model(wave_b1t_chunk.squeeze(1))[0]
                if logits_chunk_btc.shape[1] != logits_sz:
                    # wav2vec2 returns 1 less time step than expected for 50hz.
                    logits_chunk_btc = torch.nn.functional.interpolate(logits_chunk_btc.permute(0, 2, 1),
                                                                       [logits_sz], mode="nearest").permute(0, 2, 1)

                if last_logits is None:
                    last_logits = logits_chunk_btc
                else:
                    # Use the overlap to finalize last_logits into text.
                    last_logits[:, -logits_overlap_sz:] = (last_logits[:, -logits_overlap_sz:] + logits_chunk_btc[:,
                                                                                                 :logits_overlap_sz]) / 2
                    logit_chunks.append(last_logits)
                    last_logits = logits_chunk_btc[:, logits_overlap_sz:]
        logit_chunks.append(last_logits)
        logits_btc = torch.cat(logit_chunks, dim=1)
        return self.processor.batch_decode(torch.argmax(logits_btc, dim=-1)), logits_btc

    def logits_to_text_with_timings(self, logits_tc):
        """
        Returns text and the start time for each character in text. The timings are quite accurate.
        """
        tokens = torch.argmax(logits_tc, dim=-1).detach().cpu()
        str_tokens = self.processor.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=False)
        result = []
        result_timings = []
        seen_pad = False
        for i, token in enumerate(str_tokens):
            if token == self.processor.tokenizer.pad_token:
                seen_pad = True
                continue
            if not result or result[-1] != token or seen_pad:
                seen_pad = False
                assert len(token) == 1, token
                result.append(token)
                result_timings.append(i / 50.0)
        return "".join(result).replace(self.processor.tokenizer.word_delimiter_token, " "), result_timings


if __name__ == '__main__':
    transcriber = Transcriber(phonetic=False)
    audio = load_audio('../data/obama.mp3', 44100)
    audio = audio.unsqueeze(0).unsqueeze(0)

    text, logits_btc = (transcriber.transcribe_batch(torch.cat([audio, ], dim=0), 44100))
    print(text)
    # Now dump the alignment to textgrid, it can be easily viewed via praat.
    import textgrid
    grid = textgrid.TextGrid()
    char_grid = textgrid.IntervalTier()
    chars, times = transcriber.logits_to_text_with_timings(logits_btc[0])
    times.append(audio.shape[-1]/44100)
    for i in range(len(chars)):
        char_grid.add(minTime=times[i], maxTime=times[i+1], mark=chars[i])
    grid.append(char_grid)
    grid.write(open("../data/obama.textgrid", "w"))

