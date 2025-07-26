import os
import argparse
import requests
import sys

# Check if tqdm is available for progress bars
try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def download_file(url, destination):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Where to save the file
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024*1024  # 1 MB
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Show download progress
        print(f"Downloading {url} to {destination}")
        print(f"File size: {total_size / (1024*1024*1024):.2f} GB")
        
        if TQDM_AVAILABLE:
            with open(destination, 'wb') as file, tqdm.tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                file=sys.stdout
            ) as bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    bar.update(len(data))
        else:
            # Simple progress indicator without tqdm
            downloaded = 0
            with open(destination, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    downloaded += len(data)
                    percent = int(100 * downloaded / total_size)
                    sys.stdout.write(f"\rProgress: {percent}% ({downloaded/(1024*1024*1024):.2f} / {total_size/(1024*1024*1024):.2f} GB)")
                    sys.stdout.flush()
            print()  # Add a newline at the end
                
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download BioSentVec model")
    parser.add_argument("--output", type=str, 
                        help="Path to save BioSentVec model (default: ./data/models/)")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output:
        output_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "data", "models", 
                                 "BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
    
    # BioSentVec URL
    biosentvec_url = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
    
    print("This will download the BioSentVec model (~4.3GB). This may take some time.")
    print("The model will be saved to:", output_path)
    print("\nAfter downloading, you'll need to install sent2vec to use it:")
    print("pip install sent2vec")
    
    proceed = input("Do you want to proceed? (y/n): ")
    if proceed.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        return
    
    success = download_file(biosentvec_url, output_path)
    
    if success:
        print(f"\nBioSentVec model downloaded successfully to {output_path}")
        print(f"You can now use it with: python build_terminology_index.py --biosentvec {output_path}")
        
        # Check if sent2vec is installed
        try:
            import sent2vec
            print("sent2vec is already installed.")
        except ImportError:
            print("\nNow you need to install sent2vec:")
            print("pip install sent2vec")
    else:
        print("\nFailed to download BioSentVec model.")
        print("Please download it manually from:", biosentvec_url)

if __name__ == "__main__":
    main()