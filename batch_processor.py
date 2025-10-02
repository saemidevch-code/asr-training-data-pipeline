#!/usr/bin/env python3
"""
batch_processor.py
Batch processing wrapper for claude_research_optimal.py
Generates word coverage statistics per file.
"""
import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict
import subprocess

def generate_word_coverage(full_txt_path: Path) -> Dict[str, int]:
    """
    Generate word frequency count from concatenated transcript.
    Returns dict of {word: count} sorted by frequency.
    """
    if not full_txt_path.exists():
        return {}
    
    text = full_txt_path.read_text(encoding='utf-8')
    
    # Split into words, normalize to lowercase
    words = text.lower().split()
    
    # Count frequencies
    word_counts = Counter(words)
    
    # Sort by frequency (ascending - least used first), then alphabetically
    sorted_counts = dict(sorted(word_counts.items(), 
                               key=lambda x: (x[1], x[0])))
    
    return sorted_counts

def save_word_coverage_csv(word_counts: Dict[str, int], output_path: Path):
    """Save word coverage to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'count'])
        for word, count in word_counts.items():
            writer.writerow([word, count])

def process_single_file(audio_path: str, text_path: str, 
                       output_base_dir: Path,
                       cli_params: Dict) -> Tuple[bool, str]:
    """
    Process a single audio+transcript pair using claude_research_optimal.py
    
    Returns: (success: bool, message: str)
    """
    # Extract base name from audio file
    audio_file = Path(audio_path)
    base_name = audio_file.stem
    
    # Create output directory for this file
    clips_dir = output_base_dir / f"{base_name}_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    script_path = Path(__file__).parent / "claude_research_optimal.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--audio", audio_path,
        "--text", text_path,
        "--outdir", str(clips_dir),
    ]
    
    # Add optional CLI parameters
    if cli_params.get('base_end_guard_ms'):
        cmd.extend(["--base_end_guard_ms", str(cli_params['base_end_guard_ms'])])
    if cli_params.get('tail_safety_ms'):
        cmd.extend(["--tail_safety_ms", str(cli_params['tail_safety_ms'])])
    if cli_params.get('dump_asr'):
        cmd.append("--dump_asr")
    if cli_params.get('debug'):
        cmd.append("--debug")
    
    # Run processing
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if processing succeeded
        summary_path = clips_dir / "summary.json"
        if not summary_path.exists():
            return False, f"Processing failed: No summary.json generated"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        exported = summary.get('exported', 0)
        if exported == 0:
            return False, f"No segments exported (0 matches found)"
        
        # Move full.wav and full.txt to parent directory with proper naming
        full_wav_src = clips_dir / "full.wav"
        full_txt_src = clips_dir / "full.txt"
        
        full_wav_dst = output_base_dir / f"full_{base_name}.wav"
        full_txt_dst = output_base_dir / f"full_{base_name}.txt"
        
        if full_wav_src.exists():
            full_wav_src.rename(full_wav_dst)
        if full_txt_src.exists():
            full_txt_src.rename(full_txt_dst)
        
        # Generate word coverage
        if full_txt_dst.exists():
            word_counts = generate_word_coverage(full_txt_dst)
            coverage_path = output_base_dir / f"full_{base_name}_wordcoverage.csv"
            save_word_coverage_csv(word_counts, coverage_path)
        
        return True, f"Success: {exported} segments exported"
        
    except subprocess.CalledProcessError as e:
        return False, f"Processing failed: {e.stderr}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def process_batch(file_pairs: List[Tuple[str, str]], 
                 output_base_dir: str,
                 cli_params: Dict) -> Dict:
    """
    Process multiple audio+transcript pairs.
    
    Args:
        file_pairs: List of (audio_path, text_path) tuples
        output_base_dir: Base directory for all outputs
        cli_params: Dictionary of CLI parameters
    
    Returns:
        Dictionary with processing results
    """
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'total': len(file_pairs),
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    for i, (audio_path, text_path) in enumerate(file_pairs, 1):
        audio_name = Path(audio_path).name
        text_name = Path(text_path).name
        
        print(f"\n[{i}/{len(file_pairs)}] Processing: {audio_name}")
        
        success, message = process_single_file(
            audio_path, text_path, output_dir, cli_params
        )
        
        if success:
            results['successful'] += 1
            status = "✓ SUCCESS"
        else:
            results['failed'] += 1
            status = "✗ FAILED"
        
        result_entry = {
            'audio': audio_name,
            'text': text_name,
            'status': status,
            'message': message
        }
        results['details'].append(result_entry)
        
        print(f"  {status}: {message}")
    
    # Save batch summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Example usage
    import argparse
    
    ap = argparse.ArgumentParser(description="Batch process audio files for ASR training")
    ap.add_argument("--pairs", nargs='+', required=True,
                   help="Pairs of audio,transcript paths (e.g., audio1.mp4,text1.txt audio2.mp4,text2.txt)")
    ap.add_argument("--output", required=True, help="Output base directory")
    ap.add_argument("--base_end_guard_ms", type=int, default=35)
    ap.add_argument("--tail_safety_ms", type=int, default=80)
    ap.add_argument("--dump_asr", action="store_true")
    ap.add_argument("--debug", action="store_true")
    
    args = ap.parse_args()
    
    # Parse file pairs
    file_pairs = []
    for pair in args.pairs:
        parts = pair.split(',')
        if len(parts) != 2:
            print(f"Error: Invalid pair format: {pair}")
            sys.exit(1)
        file_pairs.append((parts[0], parts[1]))
    
    # CLI parameters
    cli_params = {
        'base_end_guard_ms': args.base_end_guard_ms,
        'tail_safety_ms': args.tail_safety_ms,
        'dump_asr': args.dump_asr,
        'debug': args.debug
    }
    
    # Process batch
    results = process_batch(file_pairs, args.output, cli_params)
    
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {results['successful']}/{results['total']} successful")
    print(f"{'='*60}")