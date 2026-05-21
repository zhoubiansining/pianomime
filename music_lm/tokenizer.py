from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TokenizerConfig:
    min_pitch: int = 21
    max_pitch: int = 108
    velocity_bins: int = 32
    time_step_seconds: float = 0.01
    max_time_shift_steps: int = 100
    sustain_control: int = 64

    @classmethod
    def from_dict(cls, values: dict) -> "TokenizerConfig":
        return cls(**{key: values[key] for key in asdict(cls()).keys() if key in values})


class EventTokenizer:
    """REMI-like event tokenizer for piano performance MIDI.

    The stream uses explicit time-shift, velocity, note-on, note-off, and sustain
    tokens. Durations are represented by the distance between note-on and note-off.
    """

    pad_token = 0
    bos_token = 1
    eos_token = 2
    _special_tokens = 3

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        if self.config.min_pitch > self.config.max_pitch:
            raise ValueError("min_pitch must be <= max_pitch")
        if self.config.velocity_bins <= 0:
            raise ValueError("velocity_bins must be positive")
        if self.config.time_step_seconds <= 0:
            raise ValueError("time_step_seconds must be positive")
        if self.config.max_time_shift_steps <= 0:
            raise ValueError("max_time_shift_steps must be positive")

        self.n_pitches = self.config.max_pitch - self.config.min_pitch + 1
        self.velocity_offset = self._special_tokens
        self.note_on_offset = self.velocity_offset + self.config.velocity_bins
        self.note_off_offset = self.note_on_offset + self.n_pitches
        self.time_shift_offset = self.note_off_offset + self.n_pitches
        self.sustain_on_token = self.time_shift_offset + self.config.max_time_shift_steps
        self.sustain_off_token = self.sustain_on_token + 1
        self.vocab_size = self.sustain_off_token + 1

    def to_config_dict(self) -> dict:
        return asdict(self.config)

    def velocity_token(self, velocity: int) -> int:
        velocity = max(1, min(127, int(velocity)))
        bin_id = min(
            self.config.velocity_bins - 1,
            int((velocity - 1) * self.config.velocity_bins / 127),
        )
        return self.velocity_offset + bin_id

    def note_on_token(self, pitch: int) -> int:
        return self.note_on_offset + self._pitch_index(pitch)

    def note_off_token(self, pitch: int) -> int:
        return self.note_off_offset + self._pitch_index(pitch)

    def time_shift_token(self, steps: int) -> int:
        if not 1 <= steps <= self.config.max_time_shift_steps:
            raise ValueError(
                f"time-shift steps must be in [1, {self.config.max_time_shift_steps}]"
            )
        return self.time_shift_offset + steps - 1

    def encode_midi_file(self, path: str | Path, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        from note_seq import midi_io

        seq = midi_io.midi_file_to_note_sequence(str(path))
        return self.encode_note_sequence(seq, add_bos=add_bos, add_eos=add_eos)

    def encode_note_sequence(self, seq, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        events: list[tuple[float, int, str, int]] = []
        for note in seq.notes:
            pitch = int(note.pitch)
            if not self.config.min_pitch <= pitch <= self.config.max_pitch:
                continue
            velocity = int(note.velocity) if note.velocity else 64
            events.append((float(note.start_time), 1, "velocity", velocity))
            events.append((float(note.start_time), 2, "note_on", pitch))
            events.append((float(note.end_time), 0, "note_off", pitch))

        for cc in getattr(seq, "control_changes", []):
            if int(cc.control_number) != self.config.sustain_control:
                continue
            kind = "sustain_on" if int(cc.control_value) >= 64 else "sustain_off"
            events.append((float(cc.time), 3, kind, 0))

        events.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return self._encode_sorted_events(events, add_bos=add_bos, add_eos=add_eos)

    def encode_midi_messages(
        self,
        messages: Sequence[object],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        events: list[tuple[float, int, str, int]] = []
        for message in messages:
            kind = message.__class__.__name__
            if kind == "NoteOn":
                note = int(getattr(message, "note"))
                if self.config.min_pitch <= note <= self.config.max_pitch:
                    events.append((float(getattr(message, "time")), 1, "velocity", int(getattr(message, "velocity"))))
                    events.append((float(getattr(message, "time")), 2, "note_on", note))
            elif kind == "NoteOff":
                note = int(getattr(message, "note"))
                if self.config.min_pitch <= note <= self.config.max_pitch:
                    events.append((float(getattr(message, "time")), 0, "note_off", note))
            elif kind == "SustainOn":
                events.append((float(message.time), 3, "sustain_on", 0))
            elif kind == "SustainOff":
                events.append((float(message.time), 3, "sustain_off", 0))
        events.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return self._encode_sorted_events(events, add_bos=add_bos, add_eos=add_eos)

    def token_name(self, token: int) -> str:
        token = int(token)
        if token == self.pad_token:
            return "PAD"
        if token == self.bos_token:
            return "BOS"
        if token == self.eos_token:
            return "EOS"
        if self.velocity_offset <= token < self.note_on_offset:
            return f"VELOCITY_{token - self.velocity_offset}"
        if self.note_on_offset <= token < self.note_off_offset:
            return f"NOTE_ON_{self.config.min_pitch + token - self.note_on_offset}"
        if self.note_off_offset <= token < self.time_shift_offset:
            return f"NOTE_OFF_{self.config.min_pitch + token - self.note_off_offset}"
        if self.time_shift_offset <= token < self.sustain_on_token:
            return f"TIME_SHIFT_{token - self.time_shift_offset + 1}"
        if token == self.sustain_on_token:
            return "SUSTAIN_ON"
        if token == self.sustain_off_token:
            return "SUSTAIN_OFF"
        return f"UNKNOWN_{token}"

    def _pitch_index(self, pitch: int) -> int:
        pitch = int(pitch)
        if not self.config.min_pitch <= pitch <= self.config.max_pitch:
            raise ValueError(
                f"pitch must be in [{self.config.min_pitch}, {self.config.max_pitch}], got {pitch}"
            )
        return pitch - self.config.min_pitch

    def _encode_sorted_events(
        self,
        events: Iterable[tuple[float, int, str, int]],
        add_bos: bool,
        add_eos: bool,
    ) -> list[int]:
        tokens: list[int] = [self.bos_token] if add_bos else []
        current_time = 0.0
        for event_time, _, kind, value in events:
            delta = max(0.0, float(event_time) - current_time)
            self._append_time_shift(tokens, delta)
            current_time = max(current_time, float(event_time))
            if kind == "velocity":
                tokens.append(self.velocity_token(value))
            elif kind == "note_on":
                tokens.append(self.note_on_token(value))
            elif kind == "note_off":
                tokens.append(self.note_off_token(value))
            elif kind == "sustain_on":
                tokens.append(self.sustain_on_token)
            elif kind == "sustain_off":
                tokens.append(self.sustain_off_token)
        if add_eos:
            tokens.append(self.eos_token)
        return tokens

    def _append_time_shift(self, tokens: list[int], seconds: float) -> None:
        steps = int(round(seconds / self.config.time_step_seconds))
        while steps > 0:
            shift = min(steps, self.config.max_time_shift_steps)
            tokens.append(self.time_shift_token(shift))
            steps -= shift
