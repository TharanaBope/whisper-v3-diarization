"""Command-line interface for audio transcription and diarization."""

import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .core.audio_processor import AudioProcessor
from .utils.logger import setup_logging
from .config.settings import AppConfig

# Initialize CLI app and console
app = typer.Typer(
    name="audio-transcription",
    help="Audio transcription and speaker diarization CLI tool using Whisper and WhisperX",
    add_completion=False
)
console = Console()

# Global configuration
config = AppConfig()

@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO", 
        "--log-level", 
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace token for speaker diarization"
    )
):
    """Initialize the application with global settings."""
    setup_logging(log_level)
    
    if hf_token:
        config.hf_token = hf_token
    elif not config.hf_token:
        console.print(
            "⚠️  [yellow]Warning: No HuggingFace token provided. "
            "Speaker diarization will be disabled.[/yellow]"
        )


@app.command("transcribe")
def transcribe_audio(
    input_files: List[Path] = typer.Argument(
        ..., 
        help="Audio files to transcribe",
        exists=True,
        file_okay=True,
        readable=True
    ),
    output_dir: Path = typer.Option(
        "./transcriptions",
        "--output-dir", "-o",
        help="Directory to save transcription results"
    ),
    model_size: str = typer.Option(
        "large-v3",
        "--model", "-m",
        help="Whisper model size (tiny, base, small, medium, large-v2, large-v3)"
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Audio language (auto-detect if not specified)"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Processing device (auto, cpu, cuda)"
    )
):
    """Transcribe audio files using Whisper."""
    
    # Validate inputs
    _validate_model_size(model_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = AudioProcessor(
        model_size=model_size,
        device=device,
        hf_token=config.hf_token
    )
    
    # Process files with progress indication
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        
        task = progress.add_task("Transcribing files...", total=len(input_files))
        
        for file_path in input_files:
            progress.update(task, description=f"Processing {file_path.name}")
            
            try:
                result = processor.transcribe_file(
                    file_path=file_path,
                    language=language,
                    output_dir=output_dir
                )
                
                if result["success"]:
                    console.print(f"✅ [green]Successfully transcribed {file_path.name}[/green]")
                else:
                    console.print(f"❌ [red]Failed to transcribe {file_path.name}: {result['error']}[/red]")
                    
            except Exception as e:
                console.print(f"❌ [red]Error processing {file_path.name}: {str(e)}[/red]")
            
            progress.advance(task)
    
    console.print("\n🎉 [green]Transcription complete![/green]")


@app.command("diarize")
def diarize_audio(
    input_files: List[Path] = typer.Argument(
        ..., 
        help="Audio files to diarize",
        exists=True,
        file_okay=True,
        readable=True
    ),
    output_dir: Path = typer.Option(
        "./diarizations",
        "--output-dir", "-o",
        help="Directory to save diarization results"
    ),
    model_size: str = typer.Option(
        "large-v3",
        "--model", "-m",
        help="Whisper model size for transcription"
    ),
    min_speakers: Optional[int] = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers (auto-detect if not specified)"
    ),
    max_speakers: Optional[int] = typer.Option(
        None,
        "--max-speakers", 
        help="Maximum number of speakers (auto-detect if not specified)"
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Audio language (auto-detect if not specified)"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Processing device (auto, cpu, cuda)"
    )
):
    """Transcribe with speaker diarization using WhisperX."""
    
    if not config.hf_token:
        console.print("❌ [red]HuggingFace token required for speaker diarization![/red]")
        console.print("Set HF_TOKEN environment variable or use --hf-token option.")
        raise typer.Exit(1)
    
    # Validate inputs
    _validate_model_size(model_size)
    _validate_speaker_counts(min_speakers, max_speakers)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = AudioProcessor(
        model_size=model_size,
        device=device,
        hf_token=config.hf_token
    )
    
    # Process files with progress indication
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        
        task = progress.add_task("Processing with diarization...", total=len(input_files))
        
        for file_path in input_files:
            progress.update(task, description=f"Diarizing {file_path.name}")
            
            try:
                result = processor.diarize_file(
                    file_path=file_path,
                    language=language,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    output_dir=output_dir
                )
                
                if result["success"]:
                    console.print(f"✅ [green]Successfully processed {file_path.name}[/green]")
                    console.print(f"   Speakers detected: {result.get('num_speakers', 'Unknown')}")
                else:
                    console.print(f"❌ [red]Failed to process {file_path.name}: {result['error']}[/red]")
                    
            except Exception as e:
                console.print(f"❌ [red]Error processing {file_path.name}: {str(e)}[/red]")
            
            progress.advance(task)
    
    console.print("\n🎉 [green]Diarization complete![/green]")


@app.command("process")
def process_full(
    input_files: List[Path] = typer.Argument(
        ..., 
        help="Audio files to process with full pipeline",
        exists=True,
        file_okay=True,
        readable=True
    ),
    transcription_dir: Path = typer.Option(
        "./transcriptions",
        "--transcription-dir",
        help="Directory to save transcription results"
    ),
    diarization_dir: Path = typer.Option(
        "./diarizations",
        "--diarization-dir", 
        help="Directory to save diarization results"
    ),
    model_size: str = typer.Option(
        "large-v3",
        "--model", "-m",
        help="Whisper model size"
    ),
    min_speakers: Optional[int] = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers"
    ),
    max_speakers: Optional[int] = typer.Option(
        None,
        "--max-speakers",
        help="Maximum number of speakers"
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language", "-l",
        help="Audio language"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Processing device"
    )
):
    """Process audio files with both transcription and diarization."""
    
    # Create output directories
    transcription_dir.mkdir(parents=True, exist_ok=True)
    diarization_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    _validate_model_size(model_size)
    if min_speakers or max_speakers:
        _validate_speaker_counts(min_speakers, max_speakers)
    
    # Initialize processor
    processor = AudioProcessor(
        model_size=model_size,
        device=device,
        hf_token=config.hf_token
    )
    
    # Process files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        
        task = progress.add_task("Full processing pipeline...", total=len(input_files) * 2)
        
        for file_path in input_files:
            # Transcription
            progress.update(task, description=f"Transcribing {file_path.name}")
            transcribe_result = processor.transcribe_file(
                file_path=file_path,
                language=language,
                output_dir=transcription_dir
            )
            progress.advance(task)
            
            # Diarization (if HF token available)
            if config.hf_token:
                progress.update(task, description=f"Diarizing {file_path.name}")
                diarize_result = processor.diarize_file(
                    file_path=file_path,
                    language=language,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    output_dir=diarization_dir
                )
                progress.advance(task)
                
                # Report results
                if transcribe_result["success"] and diarize_result["success"]:
                    console.print(f"✅ [green]Fully processed {file_path.name}[/green]")
                    console.print(f"   Speakers: {diarize_result.get('num_speakers', 'Unknown')}")
                else:
                    console.print(f"⚠️  [yellow]Partial processing for {file_path.name}[/yellow]")
            else:
                progress.advance(task)  # Skip diarization step
                if transcribe_result["success"]:
                    console.print(f"✅ [green]Transcribed {file_path.name} (diarization skipped - no HF token)[/green]")
    
    console.print("\n🎉 [green]Full processing complete![/green]")


def _validate_model_size(model_size: str):
    """Validate Whisper model size."""
    valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    if model_size not in valid_models:
        console.print(f"❌ [red]Invalid model size: {model_size}[/red]")
        console.print(f"Valid options: {', '.join(valid_models)}")
        raise typer.Exit(1)


def _validate_speaker_counts(min_speakers: Optional[int], max_speakers: Optional[int]):
    """Validate speaker count parameters."""
    if min_speakers is not None and min_speakers < 1:
        console.print("❌ [red]Minimum speakers must be at least 1[/red]")
        raise typer.Exit(1)
    
    if max_speakers is not None and max_speakers < 1:
        console.print("❌ [red]Maximum speakers must be at least 1[/red]")
        raise typer.Exit(1)
    
    if (min_speakers is not None and max_speakers is not None and 
        min_speakers > max_speakers):
        console.print("❌ [red]Minimum speakers cannot exceed maximum speakers[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()