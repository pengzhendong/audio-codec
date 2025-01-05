# Copyright (c) 2025, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn.functional as F
from modelscope import model_file_download
from moshi.models.loaders import FRAME_RATE, get_mimi, SAMPLE_RATE


class Mimi:
    def __init__(self, model_id: str = "AI-ModelScope", device: str = "cpu"):
        model_path = model_file_download(model_id, "tokenizer-e351c8d8-checkpoint125.safetensors")
        self.device = device
        self.model = get_mimi(model_path, device)

    @torch.inference_mode()
    def streaming_encode(self, audio: torch.Tensor):
        B, C, num_chunks, _ = audio.shape
        with self.model.streaming(B):
            for i in range(num_chunks):
                chunk = audio[..., i, :].to(self.device)
                yield self.model.encode(chunk)

    @torch.inference_mode()
    def non_streaming_encode(self, audio: torch.Tensor, batch_duration: float):
        B, C, num_chunks, chunk_size = audio.shape
        chunks = audio.view(B * num_chunks, C, chunk_size)
        batch_chunks = B * int(batch_duration * SAMPLE_RATE / B // chunk_size)
        for i in range(num_chunks):
            chunk = chunks[i : batch_chunks * num_chunks : num_chunks].to(self.device)
            yield self.model.encode(chunk)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, audio_lens: torch.Tensor = None, batch_duration: float = None, chunk_duration: float = 12, streaming: bool = True):
        B, C, T = audio.shape
        assert C == 1, "Mimi only supports mono audio."
        chunk_size = math.floor(chunk_duration * SAMPLE_RATE)
        num_chunks = (T - 1) // chunk_size + 1
        pad_size = int(num_chunks * chunk_size - T)
        if pad_size > 0:
            audio = F.pad(audio, pad=(0, pad_size), mode="constant", value=0)
        audio = audio.reshape(B, C, num_chunks, chunk_size)
        if streaming:
            codes = self.streaming_encode(audio)
        else:
            batch_duration = batch_duration or B * chunk_duration
            codes = self.non_streaming_encode(audio, batch_duration)
        codes = torch.cat(list(codes), dim=-1)
        return codes[..., :-int(pad_size * FRAME_RATE // SAMPLE_RATE)] if pad_size > 0 else codes

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, chunk_duration: float = 12):
        B, _, T = codes.shape
        chunk_size = math.floor(chunk_duration * FRAME_RATE)
        num_chunks = (T - 1) // chunk_size + 1
        pad_size = int(num_chunks * chunk_size - T)
        if pad_size > 0:
            codes = F.pad(codes, (0, pad_size), "constant", 0)

        with self.model.streaming(B):
            audio = []
            for i in range(0, T, chunk_size):
                chunk = codes[..., i : i + chunk_size].to(self.device)
                audio.append(self.model.decode(chunk))
        audio = torch.cat(audio, dim=-1)
        return audio[..., :-int(pad_size * SAMPLE_RATE // FRAME_RATE)] if pad_size > 0 else audio
