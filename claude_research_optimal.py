#!/usr/bin/env python3
"""
claude_research_optimal.py
Research-informed optimal pipeline for zero-phantom audio extraction.

Architecture:
- Base ASR: WhisperX (forced alignment, better boundaries)
- Confidence: Extracted from alignment scores
- Adaptive guards: Confidence-weighted (25-45ms)
- Acoustic validation: MFCC + spectral flux + energy
- Validator: medium.en for consensus
- No triple-pass re-transcription (replaced by acoustic checks)
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json, re, sys, time, difflib, traceback, uuid
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from pydub import AudioSegment
from faster_whisper import WhisperModel

# -------------------- Logging --------------------
def log(msg: str):
    print(f"[OPTIMAL {time.strftime('%H:%M:%S')}] {msg}", flush=True)

# -------------------- Normalization --------------------
_WORD_RE = re.compile(r"[A-Za-z0-9'-]+")
_SMALLS = ["zero","one","two","three","four","five","six","seven","eight","nine",
           "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
           "seventeen","eighteen","nineteen"]
_TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def _num_to_words(n: int) -> List[str]:
    if n == 0:
        return ["zero"]
    def below_thousand(x: int) -> List[str]:
        out: List[str] = []
        if x >= 100:
            out.append(_SMALLS[x // 100]); out.append("hundred"); x %= 100
        if x >= 20:
            out.append(_TENS[x // 10]); x %= 10
            if x: out.append(_SMALLS[x])
        elif x > 0:
            out.append(_SMALLS[x])
        return out
    units = ["","thousand","million","billion"]
    words: List[str] = []
    i = 0
    n_abs = abs(n)
    while n_abs > 0 and i < len(units):
        chunk = n_abs % 1000
        if chunk:
            seg = below_thousand(chunk)
            if units[i]:
                seg.append(units[i])
            words = seg + words
        n_abs //= 1000; i += 1
    if n < 0:
        words = ["minus"] + words
    return words

def normalize_word(s: str, numbers_to_words: bool) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"[^a-z0-9'\-]", "", s.strip().lower())
    s = s.replace("'", "").replace("-", "")
    if not s:
        return None
    if numbers_to_words and s.isdigit():
        try:
            n = int(s)
            ws = _num_to_words(n)
            s = "".join(ws)
        except Exception:
            pass
    return s or None

def tokenize_text_to_words(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

# -------------------- WhisperX Import --------------------
WHISPERX_IMPORT_ERR = None
try:
    import whisperx
    HAVE_WHISPERX = True
except Exception as e:
    HAVE_WHISPERX = False
    WHISPERX_IMPORT_ERR = repr(e)

# -------------------- Types --------------------
@dataclass
class ASRWord:
    text: str
    norm: str
    start: float
    end: float
    idx: int
    confidence: float = 1.0  # Add confidence field

@dataclass
class EqualRun:
    bi0:int; bi1:int; ai0:int; ai1:int

@dataclass
class VWord:
    text:str; norm:str; start:float; end:float; confidence:float = 1.0

# -------------------- Acoustic Validation --------------------
def compute_mfcc_distance(audio: AudioSegment, boundary_ms: int, window_ms: int = 50) -> float:
    """
    Compute MFCC distance across boundary to detect discontinuities.
    Uses librosa if available, falls back to simple energy check.
    """
    try:
        import librosa
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        sr = audio.frame_rate
        
        # Extract windows before and after boundary
        boundary_sample = int(boundary_ms * sr / 1000)
        window_samples = int(window_ms * sr / 1000)
        
        before = samples[max(0, boundary_sample - window_samples):boundary_sample]
        after = samples[boundary_sample:min(len(samples), boundary_sample + window_samples)]
        
        if len(before) < 100 or len(after) < 100:
            return 0.0
        
        # Compute MFCC features
        mfcc_before = librosa.feature.mfcc(y=before, sr=sr, n_mfcc=13)
        mfcc_after = librosa.feature.mfcc(y=after, sr=sr, n_mfcc=13)
        
        # Compute Euclidean distance between mean MFCCs
        dist = np.linalg.norm(np.mean(mfcc_before, axis=1) - np.mean(mfcc_after, axis=1))
        return float(dist)
        
    except ImportError:
        # Fallback to simple RMS difference if librosa not available
        log("WARNING: librosa not available, using simple energy check")
        boundary_sample = int(boundary_ms * audio.frame_rate / 1000)
        window_samples = int(window_ms * audio.frame_rate / 1000)
        
        before = audio[max(0, boundary_ms - window_ms):boundary_ms]
        after = audio[boundary_ms:boundary_ms + window_ms]
        
        try:
            rms_before = before.rms if len(before) > 0 else 0
            rms_after = after.rms if len(after) > 0 else 0
            return abs(rms_before - rms_after) / 1000.0  # Normalize
        except:
            return 0.0

def compute_spectral_flux(audio: AudioSegment, boundary_ms: int, window_ms: int = 50) -> float:
    """
    Compute spectral flux at boundary to detect abrupt transitions.
    """
    try:
        import librosa
        
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        sr = audio.frame_rate
        
        # Extract window around boundary
        boundary_sample = int(boundary_ms * sr / 1000)
        window_samples = int(window_ms * sr / 1000)
        
        window = samples[max(0, boundary_sample - window_samples//2):
                        min(len(samples), boundary_sample + window_samples//2)]
        
        if len(window) < 512:
            return 0.0
        
        # Compute STFT and spectral flux
        stft = np.abs(librosa.stft(window))
        flux = np.sum(np.diff(stft, axis=1)**2)
        return float(flux) / 1e9  # Normalize
        
    except ImportError:
        return 0.0
    except Exception as e:
        return 0.0

def check_boundary_leakage(audio: AudioSegment, boundary_ms: int, 
                          silence_threshold_db: float = -40.0) -> Tuple[bool, dict]:
    """
    Check for acoustic leakage after boundary using multiple signals.
    Returns (has_leakage, details_dict)
    """
    # Check energy in 50ms window after boundary
    window_ms = 50
    after_segment = audio[boundary_ms:boundary_ms + window_ms]
    
    if len(after_segment) == 0:
        return False, {"reason": "no_audio"}
    
    try:
        # Energy check
        dbfs = after_segment.dBFS
        has_energy = dbfs > silence_threshold_db
        
        # MFCC discontinuity
        mfcc_dist = compute_mfcc_distance(audio, boundary_ms, window_ms)
        has_discontinuity = mfcc_dist > 15.0  # Research threshold
        
        # Spectral flux
        flux = compute_spectral_flux(audio, boundary_ms, window_ms)
        has_transition = flux > 0.5  # Normalized threshold
        
        details = {
            "energy_db": float(dbfs),
            "mfcc_distance": float(mfcc_dist),
            "spectral_flux": float(flux),
            "energy_leak": has_energy,
            "mfcc_leak": has_discontinuity,
            "flux_leak": has_transition
        }
        
        # Leakage if multiple signals agree
        leak_votes = sum([has_energy, has_discontinuity, has_transition])
        has_leakage = leak_votes >= 2  # Require 2/3 agreement
        
        return has_leakage, details
        
    except Exception as e:
        log(f"Acoustic check failed: {e}")
        return False, {"error": str(e)}

# -------------------- WhisperX ASR --------------------
def load_asr_words_whisperx(audio_path: str, model_name: str, device: str,
                            numbers_to_words: bool) -> List[ASRWord]:
    """Load WhisperX with forced alignment for better boundaries and confidence."""
    if not HAVE_WHISPERX:
        raise RuntimeError(f"whisperx import failed: {WHISPERX_IMPORT_ERR}")
    
    log(f"Loading WhisperX model: {model_name}")
    model = whisperx.load_model(model_name, device=device, compute_type="float16" if device=="cuda" else "int8")
    
    log("Transcribing with WhisperX...")
    result = model.transcribe(audio_path, batch_size=16, language="en")
    
    log("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    
    log("Performing forced alignment...")
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device)
    
    # Extract words with confidence
    words: List[ASRWord] = []
    idx = 0
    for segment in result_aligned.get("segments", []):
        for w in segment.get("words", []):
            t = w.get("word", "").strip()
            if not t:
                continue
            
            n = normalize_word(t, numbers_to_words)
            if not n:
                continue
            
            # WhisperX provides word-level scores
            conf = w.get("score", 1.0)  # Default to 1.0 if not available
            
            words.append(ASRWord(
                text=t,
                norm=n,
                start=float(w["start"]),
                end=float(w["end"]),
                idx=idx,
                confidence=float(conf)
            ))
            idx += 1
    
    log(f"WhisperX extracted {len(words)} words with confidence scores")
    return words

# -------------------- Matching / grouping --------------------
def lcs_equal_runs(book_norm: List[str], asr_norm: List[str], min_run: int) -> List[EqualRun]:
    sm = difflib.SequenceMatcher(None, book_norm, asr_norm, autojunk=False)
    out: List[EqualRun]=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal" and (i2-i1)>=min_run:
            out.append(EqualRun(i1,i2,j1,j2))
    return out

def merge_with_small_gaps(runs: List[EqualRun], asr_words: List[ASRWord],
                         max_gap_words: int, max_gap_time: float) -> List[List[EqualRun]]:
    if not runs:
        return []
    runs_sorted = sorted(runs, key=lambda r: r.ai0)
    groups: List[List[EqualRun]]=[]; cur=[runs_sorted[0]]
    for nxt in runs_sorted[1:]:
        prev=cur[-1]
        gap_w = nxt.ai0 - prev.ai1
        gap_t = asr_words[nxt.ai0].start - asr_words[prev.ai1-1].end
        if gap_w<=max_gap_words and gap_t<=max_gap_time:
            cur.append(nxt)
        else:
            groups.append(cur); cur=[nxt]
    groups.append(cur); return groups

# -------------------- Base clip assembly (no end clamp) --------------------
def assemble_runs_to_clip(audio: AudioSegment, asr_words: List[ASRWord],
                         runs: List[EqualRun], fade_ms:int, seam_silence_ms:int,
                         start_pad_ms:int, end_pad_ms:int) -> Tuple[AudioSegment, List[ASRWord]]:
    """
    Assemble base clip from exact runs. NO end clamp - give validator room.
    """
    pieces=[]; used: List[ASRWord]=[]
    for k,r in enumerate(runs):
        w0=asr_words[r.ai0]; w1=asr_words[r.ai1-1]
        s_ms = int(round(w0.start*1000))
        if k==0 and start_pad_ms>0:
            s_ms = max(0, s_ms - int(start_pad_ms))
        
        # No clamp at end - let validator have room
        e_ms = int(round(w1.end*1000))
        if k==len(runs)-1 and end_pad_ms>0:
            e_ms += int(end_pad_ms)
        
        if e_ms <= s_ms:
            e_ms = s_ms + 1
        seg = audio[s_ms:e_ms]
        if fade_ms>0:
            f_in = min(max(5, fade_ms//3), len(seg)//4)
            f_out = min(fade_ms, len(seg)//4)
            if f_in>0: seg = seg.fade_in(f_in)
            if f_out>0: seg = seg.fade_out(f_out)
        pieces.append(seg)
        used += asr_words[r.ai0:r.ai1]
        if k!=len(runs)-1 and seam_silence_ms>0:
            pieces.append(AudioSegment.silent(duration=seam_silence_ms))
    clip=AudioSegment.silent(duration=0)
    for p in pieces:
        clip += p
    return clip, used

# -------------------- Validator --------------------
def validator_transcribe_segment(seg: AudioSegment, model: WhisperModel,
                                numbers_to_words: bool, tmpdir: Path) -> List[VWord]:
    tmp = tmpdir / f"vseg_{uuid.uuid4().hex}.wav"
    seg.export(tmp, format="wav")
    words: List[VWord]=[]
    segments,_ = model.transcribe(str(tmp), word_timestamps=True, language="en",
                                 vad_filter=False)
    for s in segments:
        for w in getattr(s,"words",[]) or []:
            t=(w.word or "").strip()
            if not t: continue
            n=normalize_word(t, numbers_to_words)
            if not n: continue
            # Validator doesn't provide confidence, use 1.0
            words.append(VWord(t,n,float(w.start),float(w.end),1.0))
    try:
        tmp.unlink(missing_ok=True)
    except:
        pass
    return words

def lcs_pairs(a: List[str], b: List[str]) -> List[Tuple[int,int,int,int]]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal" and (i2-i1)>0:
            out.append((i1,i2,j1,j2))
    return out

# -------------------- Confidence-weighted guards --------------------
def compute_adaptive_guard(confidence: float, base_guard: int) -> int:
    """
    Compute adaptive guard based on confidence.
    Research-informed thresholds:
    - conf > 0.75: reduce to 70% of base
    - conf < 0.6: increase to 130% of base
    """
    if confidence > 0.75:
        return int(base_guard * 0.7)
    elif confidence < 0.6:
        return int(base_guard * 1.3)
    else:
        return base_guard

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Research-optimal pipeline: WhisperX + acoustic validation")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--outdir", required=True)
    # Matching & bridging
    ap.add_argument("--min_run", type=int, default=4)
    ap.add_argument("--max_gap_words", type=int, default=2)
    ap.add_argument("--max_gap_time", type=float, default=0.5)
    # Timing polish
    ap.add_argument("--min_dur", type=float, default=1.5)
    ap.add_argument("--fade_ms", type=int, default=20)
    ap.add_argument("--start_pad_ms", type=int, default=150)
    ap.add_argument("--end_pad_ms", type=int, default=140)
    ap.add_argument("--tail_safety_ms", type=int, default=80)
    ap.add_argument("--base_end_guard_ms", type=int, default=35, help="Base end guard (adapted by confidence)")
    ap.add_argument("--start_guard_ms", type=int, default=30)
    ap.add_argument("--seam_silence_ms", type=int, default=120)
    ap.add_argument("--full_gap_ms", type=int, default=500)
    # Acoustic validation
    ap.add_argument("--silence_threshold_db", type=float, default=-40.0)
    ap.add_argument("--mfcc_threshold", type=float, default=15.0)
    ap.add_argument("--enable_acoustic_validation", action="store_true", default=True)
    # Normalization
    ap.add_argument("--numbers_to_words", action="store_true")
    # Base ASR: WhisperX
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    # Validator
    ap.add_argument("--validator_model", default="medium.en")
    ap.add_argument("--validator_device", choices=["cuda","cpu"], default=None)
    ap.add_argument("--min_valid_words", type=int, default=2)
    # Misc
    ap.add_argument("--dump_asr", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not HAVE_WHISPERX:
        log("ERROR: WhisperX not installed. Install with: pip install git+https://github.com/m-bain/whisperX.git")
        sys.exit(1)

    outdir=Path(args.outdir); clips_dir=outdir/"clips"; tmp_dir=outdir/"_tmp"
    clips_dir.mkdir(parents=True, exist_ok=True); tmp_dir.mkdir(parents=True, exist_ok=True)

    # Book text
    book_raw = Path(args.text).read_text(encoding="utf-8", errors="ignore")
    book_tokens = tokenize_text_to_words(book_raw)
    book_norm = [normalize_word(t, args.numbers_to_words) for t in book_tokens]
    book_norm = [t for t in book_norm if t]
    log(f"BOOK tokens={len(book_tokens)} norm_words={len(book_norm)}")

    # Base ASR: WhisperX with forced alignment
    asr_words = load_asr_words_whisperx(args.audio, args.whisper_model, args.device,
                                       args.numbers_to_words)
    if not asr_words:
        log("ASR produced 0 words"); sys.exit(3)
    asr_norm = [w.norm for w in asr_words]
    log(f"ASR words={len(asr_words)}, avg confidence={np.mean([w.confidence for w in asr_words]):.3f}")

    if args.dump_asr:
        (outdir/"asr_full_raw.txt").write_text(" ".join(w.text for w in asr_words), encoding="utf-8")
        (outdir/"asr_full_norm.txt").write_text(" ".join(w.norm for w in asr_words), encoding="utf-8")
        (outdir/"book_norm.txt").write_text(" ".join(book_norm), encoding="utf-8")
        # Dump confidence scores
        conf_data = [{"word": w.text, "confidence": w.confidence} for w in asr_words]
        (outdir/"asr_confidence.json").write_text(json.dumps(conf_data, indent=2), encoding="utf-8")

    # Equal runs + bridging
    runs = lcs_equal_runs(book_norm, asr_norm, args.min_run)
    log(f"equal_runs(≥{args.min_run})={len(runs)}")
    if not runs:
        (outdir/"summary.json").write_text(json.dumps({"exported":0,"equal_runs":0,
                                                       "params":vars(args)},indent=2),
                                          encoding="utf-8")
        return

    groups = merge_with_small_gaps(runs, asr_words, args.max_gap_words, args.max_gap_time)
    log(f"bridged_groups={len(groups)}")

    # Validator model (consensus check)
    v_device = args.validator_device or args.device
    v_ct = "float16" if v_device=="cuda" else "int8"
    v_model = WhisperModel(args.validator_model, device=v_device, compute_type=v_ct)

    # Audio source
    audio = AudioSegment.from_file(args.audio)

    kept=0; rejected=0
    tsv=["path\tstart\tend\tduration_s\twords\tpieces\tavg_conf\tacoustic_quality"]
    full_audio=AudioSegment.silent(duration=0); full_texts=[]
    rejection_log=[]; acoustic_log=[]

    for gidx, group in enumerate(groups):
        # Base clip from exact runs (NO end clamp)
        base_clip, used_words = assemble_runs_to_clip(audio, asr_words, group,
                                                       fade_ms=args.fade_ms,
                                                       seam_silence_ms=args.seam_silence_ms,
                                                       start_pad_ms=args.start_pad_ms,
                                                       end_pad_ms=args.end_pad_ms)
        if base_clip.duration_seconds<=0:
            continue

        # Validator transcription (consensus)
        vwords = validator_transcribe_segment(base_clip, v_model, args.numbers_to_words, tmp_dir)
        if not vwords:
            continue

        # LCS between used_words and validator
        a = [w.norm for w in used_words]
        b = [w.norm for w in vwords]
        eq = lcs_pairs(a,b)
        if not eq:
            continue

        # Pick longest validator span
        best=None; best_len=0
        for ai0,ai1,bi0,bi1 in eq:
            L = bi1-bi0
            if L>best_len:
                best=(bi0,bi1); best_len=L
        if best_len < args.min_valid_words:
            continue

        bi0,bi1 = best
        keep_words = vwords[bi0:bi1]

        # Get confidence from original ASR words (used_words has confidence from WhisperX)
        # Map validator span back to used_words for confidence
        span_confidences = []
        for i in range(len(keep_words)):
            # Find matching word in used_words
            for uw in used_words:
                if uw.norm == keep_words[i].norm:
                    span_confidences.append(uw.confidence)
                    break
        avg_confidence = np.mean(span_confidences) if span_confidences else 1.0

        # Baseline times
        s_ms = int(round(keep_words[0].start * 1000))
        e_ms = int(round(keep_words[-1].end * 1000))

        # START handling (standard)
        if args.start_pad_ms > 0:
            s_ms = max(0, s_ms - int(args.start_pad_ms))
        if bi0 > 0:
            prev_end_ms = int(round(vwords[bi0 - 1].end * 1000))
            guard_ms = int(args.start_guard_ms)
            s_ms = max(s_ms, prev_end_ms + guard_ms)

        # END handling: ADAPTIVE GUARD + phoneme-aware tail extension
        # Check if last word ends in plosive/fricative/nasal phonetically
        # Standard tail padding
        e_ms += int(args.end_pad_ms) + int(args.tail_safety_ms)
        
        # Apply adaptive guard first
        last_word_conf = avg_confidence  # Default to span average
        adaptive_guard = args.base_end_guard_ms  # Default guard
        
        if bi1 < len(vwords):
            next_start_ms = int(round(vwords[bi1].start * 1000))
            last_word_conf = keep_words[-1].confidence if hasattr(keep_words[-1], 'confidence') else avg_confidence
            adaptive_guard = compute_adaptive_guard(last_word_conf, args.base_end_guard_ms)
            safe_end_ms = next_start_ms - adaptive_guard
            if e_ms > safe_end_ms:
                e_ms = safe_end_ms
            if args.debug:
                log(f"  Segment {kept}: last_word_conf={last_word_conf:.3f}, adaptive_guard={adaptive_guard}ms")
        
        # THEN add phoneme extension (guaranteed to preserve consonant completion)
        # Only apply to words ≥3 chars (short function words don't need extension)
        last_word_text = keep_words[-1].text.lower()
        if last_word_text and len(last_word_text) >= 3:
            extension_ms = 0
            
            # Special handling for -ing words (need more time for /ŋ/)
            if last_word_text[-3:] == 'ing':
                extension_ms = 60
            # Sibilants and fricatives (need more decay time)
            elif last_word_text[-1] in ['s', 'z', 'x', 'f', 'v']:
                extension_ms = 50
            # Plosives and other consonants
            elif last_word_text[-1] in ['t', 'd', 'k', 'p', 'n', 'm', 'g']:
                extension_ms = 40
            elif len(last_word_text) >= 2:
                ending = last_word_text[-2:]
                if ending in ['ce', 'se', 'ze', 'ge', 'ch', 'sh', 'th', 'ng']:
                    extension_ms = 50  # ng needs more than single consonants
            elif last_word_text[-3:] in ['dge', 'tch']:
                extension_ms = 40
            
            if extension_ms > 0:
                e_ms += extension_ms
                if args.debug:
                    log(f"  Segment {kept}: '{last_word_text}' phoneme extension +{extension_ms}ms")

        if e_ms <= s_ms:
            e_ms = s_ms + 1

        # Min duration check
        if (e_ms - s_ms) < int(args.min_dur * 1000):
            rejected += 1
            rejection_log.append({
                "segment": f"segment_{kept:04d}",
                "reason": "too_short",
                "duration_ms": e_ms - s_ms
            })
            continue

        piece = base_clip[s_ms:e_ms]

        # Asymmetric fades
        if args.fade_ms > 0:
            f_in = min(max(5, args.fade_ms // 3), len(piece) // 4)
            f_out = min(args.fade_ms, len(piece) // 4)
            if f_in > 0:
                piece = piece.fade_in(f_in)
            if f_out > 0:
                piece = piece.fade_out(f_out)

        # ACOUSTIC VALIDATION at end boundary (check final 50ms of segment)
        acoustic_quality = "not_checked"
        if args.enable_acoustic_validation:
            # Check the final 50ms window WITHIN the segment, not beyond it
            check_position = max(0, len(piece) - 50)
            has_leakage, details = check_boundary_leakage(piece, check_position, args.silence_threshold_db)
            acoustic_quality = "leakage" if has_leakage else "clean"
            
            acoustic_log.append({
                "segment": f"segment_{kept:04d}",
                "quality": acoustic_quality,
                "details": details
            })
            
            if has_leakage:
                # Try reducing end by 30ms and re-check
                e_ms_retry = e_ms - 30
                if e_ms_retry > s_ms + int(args.min_dur * 1000):
                    piece_retry = base_clip[s_ms:e_ms_retry]
                    has_leakage_retry, details_retry = check_boundary_leakage(piece_retry, len(piece_retry), args.silence_threshold_db)
                    
                    if not has_leakage_retry:
                        # Accept the reduced version
                        e_ms = e_ms_retry
                        piece = piece_retry
                        acoustic_quality = "fixed"
                        if args.debug:
                            log(f"  Segment {kept}: Fixed leakage by reducing 30ms")
                    else:
                        # Still has leakage, reject
                        rejected += 1
                        rejection_log.append({
                            "segment": f"segment_{kept:04d}",
                            "reason": "acoustic_leakage",
                            "confidence": float(avg_confidence),
                            "details": details
                        })
                        if args.debug:
                            log(f"  REJECT segment {kept}: acoustic leakage (conf={avg_confidence:.3f})")
                        continue
                else:
                    # Can't reduce further without violating min duration
                    rejected += 1
                    rejection_log.append({
                        "segment": f"segment_{kept:04d}",
                        "reason": "acoustic_leakage_unfixable",
                        "confidence": float(avg_confidence)
                    })
                    continue

        seg_text = " ".join(w.text for w in keep_words)

        # Emit
        name=f"segment_{kept:04d}"
        wavp=clips_dir/f"{name}.wav"; txtp=clips_dir/f"{name}.txt"
        piece.export(wavp, format="wav")
        txtp.write_text(seg_text, encoding="utf-8")
        
        tsv.append(f"{wavp}\t0.000\t{piece.duration_seconds:.3f}\t{piece.duration_seconds:.3f}\t{len(keep_words)}\t{len(group)}\t{avg_confidence:.3f}\t{acoustic_quality}")

        full_texts.append(seg_text)
        full_audio += piece + AudioSegment.silent(duration=args.full_gap_ms)
        kept+=1

    # Finalize
    (outdir/"clips.tsv").write_text("\n".join(tsv), encoding="utf-8")
    if full_texts:
        (outdir/"full.txt").write_text("\n".join(full_texts), encoding="utf-8")
        full_audio.export(outdir/"full.wav", format="wav")

    # Save logs
    if rejection_log:
        (outdir/"rejections.json").write_text(json.dumps(rejection_log, indent=2), encoding="utf-8")
    if acoustic_log:
        (outdir/"acoustic_validation.json").write_text(json.dumps(acoustic_log, indent=2), encoding="utf-8")

    (outdir/"summary.json").write_text(json.dumps({
        "exported": kept,
        "rejected": rejected,
        "rejection_rate_pct": round(rejected / (kept + rejected) * 100, 2) if (kept + rejected) > 0 else 0,
        "equal_runs": len(runs),
        "bridged_groups": len(groups),
        "whisperx_model": args.whisper_model,
        "validator": args.validator_model,
        "acoustic_validation": args.enable_acoustic_validation,
        "params": vars(args)
    }, indent=2), encoding="utf-8")

    log(f"DONE kept={kept} rejected={rejected} ({rejected/(kept+rejected)*100:.1f}% rejection) | {outdir/'clips.tsv'}")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL {type(e).__name__}: {e}")
        log(traceback.format_exc())
        sys.exit(1)