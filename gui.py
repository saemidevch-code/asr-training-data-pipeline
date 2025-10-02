#!/usr/bin/env python3
"""
gui.py
Gradio interface for batch ASR training data extraction.
"""
import gradio as gr
from pathlib import Path
from datetime import datetime
from batch_processor import process_batch
import json

def create_output_dir(base_path: str) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def process_files(audio_files, text_files, output_base_dir,
                 base_end_guard, tail_safety, dump_asr, debug):
    """
    Process uploaded files through batch processor.
    """
    if not audio_files or not text_files:
        return "❌ Error: Please upload both audio and text files"
    
    if len(audio_files) != len(text_files):
        return f"❌ Error: Number of audio files ({len(audio_files)}) must match number of text files ({len(text_files)})"
    
    # Create output directory
    output_dir = create_output_dir(output_base_dir)
    
    # Prepare file pairs
    file_pairs = []
    for audio, text in zip(audio_files, text_files):
        audio_path = audio.name if hasattr(audio, 'name') else audio
        text_path = text.name if hasattr(text, 'name') else text
        file_pairs.append((audio_path, text_path))
    
    # CLI parameters
    cli_params = {
        'base_end_guard_ms': int(base_end_guard),
        'tail_safety_ms': int(tail_safety),
        'dump_asr': dump_asr,
        'debug': debug
    }
    
    # Process batch
    try:
        results = process_batch(file_pairs, output_dir, cli_params)
        
        # Format results
        output_text = f"""
{'='*60}
BATCH PROCESSING COMPLETE
{'='*60}

Output Directory: {output_dir}

Results:
  Total Files:      {results['total']}
  ✓ Successful:     {results['successful']}
  ✗ Failed:         {results['failed']}

Details:
"""
        
        for detail in results['details']:
            output_text += f"\n{detail['status']} {detail['audio']}\n"
            output_text += f"  → {detail['message']}\n"
        
        output_text += f"\n{'='*60}\n"
        output_text += f"Full results saved to: {output_dir}/batch_summary.json\n"
        
        # Generate word coverage summary
        coverage_files = list(Path(output_dir).glob("*_wordcoverage.csv"))
        if coverage_files:
            output_text += f"\nWord coverage files generated: {len(coverage_files)}\n"
        
        return output_text
        
    except Exception as e:
        return f"❌ Error during processing: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="ASR Training Data Extractor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ ASR Training Data Extraction Pipeline
    
    Upload matching audio and transcript files for batch processing.
    Generates clean audio clips with zero-phantom word validation.
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(
                label="Audio Files (mp4, wav, mp3)",
                file_count="multiple",
                file_types=["audio", ".mp4", ".wav", ".mp3", ".m4a"]
            )
            
            text_input = gr.File(
                label="Transcript Files (txt)",
                file_count="multiple",
                file_types=[".txt"]
            )
            
            gr.Markdown("""
            **Note:** Upload files in matching order:
            - First audio matches first transcript
            - Second audio matches second transcript
            - etc.
            """)
        
        with gr.Column():
            output_dir = gr.Textbox(
                label="Output Base Directory",
                value="C:/datasets",
                placeholder="C:/datasets"
            )
            
            gr.Markdown("### Processing Parameters")
            
            with gr.Row():
                base_guard = gr.Number(
                    label="Base End Guard (ms)",
                    value=35,
                    info="Base guard before next word (adapted by confidence)"
                )
                
                tail_safety = gr.Number(
                    label="Tail Safety (ms)",
                    value=80,
                    info="Additional safety margin at segment ends"
                )
            
            with gr.Row():
                dump_asr = gr.Checkbox(
                    label="Dump ASR Outputs",
                    value=True,
                    info="Save intermediate ASR transcriptions for debugging"
                )
                
                debug = gr.Checkbox(
                    label="Debug Mode",
                    value=False,
                    info="Verbose per-segment logging"
                )
    
    process_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
    
    output_text = gr.Textbox(
        label="Processing Results",
        lines=20,
        max_lines=30,
        show_copy_button=True
    )
    
    gr.Markdown("""
    ---
    ### Output Structure
    
    Each processed file generates:
    - `{filename}_clips/` - Individual validated segments
    - `full_{filename}.wav` - Concatenated audio
    - `full_{filename}.txt` - Concatenated transcripts
    - `full_{filename}_wordcoverage.csv` - Word frequency statistics
    
    ### Features
    - ✅ Zero-phantom word design
    - ✅ Confidence-weighted adaptive guards
    - ✅ Phoneme-aware tail extension
    - ✅ Multi-stage validation (WhisperX + validator consensus)
    - ✅ Word coverage analysis
    """)
    
    # Connect button to processing function
    process_btn.click(
        fn=process_files,
        inputs=[
            audio_input,
            text_input,
            output_dir,
            base_guard,
            tail_safety,
            dump_asr,
            debug
        ],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )