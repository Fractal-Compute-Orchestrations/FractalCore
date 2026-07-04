import os
import time
import urllib.request

url = "https://files.pythonhosted.org/packages/86/91/dedad8403e7b0036d99be4878987693b7b7f62097eb8537fa6ce62ea131c/tensorflow-2.21.0-cp313-cp313-win_amd64.whl"
filename = "tensorflow-2.21.0-cp313-cp313-win_amd64.whl"

def download():
    print(f"Starting download of: {filename}")
    while True:
        try:
            current_size = os.path.getsize(filename) if os.path.exists(filename) else 0
            req = urllib.request.Request(url)
            if current_size > 0:
                req.add_header("Range", f"bytes={current_size}-")
                print(f"Resuming download from byte {current_size}...")
            
            with urllib.request.urlopen(req) as response:
                # Fastly/PyPI will return 206 Partial Content if Range is requested
                headers = response.info()
                content_length_str = headers.get("Content-Length")
                total_size = int(content_length_str) if content_length_str else 0
                if current_size > 0:
                    total_size += current_size
                
                mode = "ab" if current_size > 0 else "wb"
                with open(filename, mode) as f:
                    chunk_size = 512 * 1024  # 512 KB chunks for robust saving
                    downloaded = current_size
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100 if total_size else 0
                        print(f"\rProgress: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.2f}%)", end="", flush=True)
            
            print("\nDownload finished successfully!")
            break
        except Exception as e:
            print(f"\nConnection dropped or error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    download()
