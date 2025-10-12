#!/usr/bin/env python3
"""
Download PDB files from HuggingFace dataset.

This script downloads PDB files from the VenusX_Motif_AlphaFold2_PDB dataset,
extracts them to a raw subdirectory, and cleans up the temporary zip file.
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from loguru import logger


def download_with_progress(url: str, filepath: Path, chunk_size: int = 8192, timeout: int = 300) -> None:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=f"Downloading {filepath.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise


def extract_and_process_pdb_files(zip_path: Path, raw_dir: Path) -> None:
    """Extract zip file and process PDB files: move to raw/ and rename to {uid}.pdb."""
    try:
        # Create temporary extraction directory
        temp_dir = raw_dir.parent / "temp_extract"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract zip file to temporary directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            
            with tqdm(
                desc=f"Extracting {zip_path.name}",
                total=len(members),
                unit='files',
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ) as pbar:
                for member in members:
                    zip_ref.extract(member, temp_dir)
                    pbar.update(1)
        
        # Find all PDB files in the extracted directory
        pdb_files = list(temp_dir.glob("**/*.pdb"))
        logger.info(f"Found {len(pdb_files)} PDB files to process")
        
        # Process PDB files: rename and move to raw directory
        processed_files = set()  # Track processed UIDs to avoid duplicates
        moved_count = 0
        duplicate_count = 0
        
        with tqdm(
            desc="Processing PDB files",
            total=len(pdb_files),
            unit='files',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            for pdb_file in pdb_files:
                # Extract UID from filename: {interpro_id}_{uid}.pdb -> {uid}.pdb
                filename = pdb_file.name
                if '_' in filename:
                    # Split by underscore and take the part after the last underscore
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        uid_part = '_'.join(parts[1:])  # Join everything after first underscore
                        uid = uid_part.replace('.pdb', '')  # Remove .pdb extension
                        
                        # Check for duplicates
                        if uid not in processed_files:
                            new_filename = f"{uid}.pdb"
                            new_path = raw_dir / new_filename
                            
                            # Move and rename file
                            pdb_file.rename(new_path)
                            processed_files.add(uid)
                            moved_count += 1
                        else:
                            duplicate_count += 1
                            logger.debug(f"Skipping duplicate UID: {uid}")
                
                pbar.update(1)
        
        logger.info(f"Processed {moved_count} PDB files, skipped {duplicate_count} duplicates")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        logger.info("Temporary extraction directory cleaned up")
                    
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file {zip_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract and process {zip_path}: {e}")
        raise


def validate_dataset_types(dataset_types: List[str], available_urls: dict) -> None:
    """Validate that requested dataset types are available."""
    available_types = list(available_urls.keys())
    invalid_types = [dt for dt in dataset_types if dt not in available_types]
    if invalid_types:
        logger.error(f"Invalid dataset types: {invalid_types}")
        logger.error(f"Available types: {available_types}")
        raise ValueError(f"Invalid dataset types: {invalid_types}")


@hydra.main(version_base=None, config_path="configs", config_name="download_pdb")
def main(cfg: DictConfig) -> None:
    """Main download function."""
    # Setup base directories (following dataset_prep.py pattern)
    plasma_path = Path(__file__).parent
    base_download_dir = plasma_path / "data" / "pdb" if cfg.output_dir is None else Path(cfg.output_dir)
    raw_dir = base_download_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to save to pdb directory
    log_file = base_download_dir / "download_pdb.log"
    logger.add(log_file, rotation="10 MB", level="INFO")
    
    logger.info(f"Starting PDB download with config: {cfg.job_name}")
    
    # Validate dataset types
    validate_dataset_types(cfg.dataset_types, cfg.download_urls)
    logger.info(f"Downloading dataset types: {cfg.dataset_types}")
    
    logger.info(f"Download directory: {base_download_dir}")
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Process each dataset type
    for dataset_type in cfg.dataset_types:
        logger.info(f"\n--- Processing {dataset_type} ---")
        
        # Get URL and filename for this dataset type
        download_url = cfg.download_urls[dataset_type]
        zip_filename = f"VenusX_{dataset_type.title()}_AlphaFold2_PDB.zip"
        zip_path = base_download_dir / zip_filename
        
        logger.info(f"Download URL: {download_url}")
        logger.info(f"Temporary zip file: {zip_path}")
        
        try:
            # Download the zip file
            logger.info(f"Starting download for {dataset_type}...")
            download_with_progress(
                download_url, 
                zip_path, 
                chunk_size=cfg.chunk_size,
                timeout=cfg.timeout
            )
            logger.success(f"Download completed: {zip_path}")
            
            # Extract and process PDB files
            logger.info(f"Starting extraction and processing for {dataset_type}...")
            extract_and_process_pdb_files(zip_path, raw_dir)
            logger.success(f"Extraction and processing completed to: {raw_dir}")
            
            # Clean up zip file
            logger.info("Cleaning up temporary files...")
            zip_path.unlink()
            logger.success("Temporary zip file removed")
            
        except Exception as e:
            logger.error(f"Download failed for {dataset_type}: {e}")
            # Clean up partial files
            if zip_path.exists():
                zip_path.unlink()
                logger.info("Cleaned up partial zip file")
            sys.exit(1)
    
    # Show final summary
    logger.info("\n--- Final Summary ---")
    extracted_files = list(raw_dir.glob("**/*"))
    logger.info(f"Total extracted files/directories: {len(extracted_files)}")
    
    # Show some example files
    pdb_files = list(raw_dir.glob("**/*.pdb"))
    if pdb_files:
        logger.info(f"Found {len(pdb_files)} PDB files")
        logger.info("Sample PDB files:")
        for pdb_file in pdb_files[:5]:  # Show first 5
            logger.info(f"  {pdb_file.relative_to(raw_dir)}")
        if len(pdb_files) > 5:
            logger.info(f"  ... and {len(pdb_files) - 5} more")
    
    logger.success("All PDB downloads completed successfully!")


if __name__ == "__main__":
    main()