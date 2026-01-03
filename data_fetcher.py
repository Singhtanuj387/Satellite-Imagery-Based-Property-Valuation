import pandas as pd
import requests
from pathlib import Path
import time
import io
from PIL import Image

CSV_FILE = "/home/devil/DL/CDC_Project/test2(test(1)).csv"
OUTPUT_DIR = Path("res_images")


METHOD = "mapbox"


MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN" 


IMAGE_SIZE = (1280, 1280)
ZOOM = 19                  

TEST_MODE = False
TEST_COUNT = 5 

def download_mapbox(property_id, lat, lon, output_dir, token):
    try:
        style = "mapbox/satellite-v9"
        width, height = IMAGE_SIZE
        
        url = (f"https://api.mapbox.com/styles/v1/{style}/static/"
               f"{lon},{lat},{ZOOM},0/{width}x{height}@2x?"
               f"access_token={token}")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            filename = f"property_{property_id}.jpg"
            filepath = output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return True, f"{len(response.content)/1024:.1f} KB"
        else:
            return False, f"HTTP {response.status_code}: {response.text[:100]}"
            
    except Exception as e:
        return False, str(e)[:100]



def main():
    print(f"\nMethod: {METHOD.upper()}")
    df = pd.read_csv(CSV_FILE)
    
    if TEST_MODE:
        df = df.head(TEST_COUNT)
        print(f"      TEST MODE: {len(df)} properties")
    else:
        print(f"      FULL MODE: {len(df)} properties")
    
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"      Output: {OUTPUT_DIR.absolute()}")
    

    if METHOD == "mapbox":
        download_func = lambda pid, lat, lon, out: download_mapbox(
            pid, lat, lon, out, MAPBOX_TOKEN)
    else:
        print(f"Invalid method: {METHOD}")
        return
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for idx, row in df.iterrows():
        property_id = row['id']
        lat = row['lat']
        lon = row['long']
        
        filename = f"property_{property_id}.jpg"
        
        if (OUTPUT_DIR / filename).exists():
            successful += 1
            print(f"[{idx+1}/{len(df)}] {filename:45s} SKIP (exists)")
            continue
        
        success, message = download_func(property_id, lat, lon, OUTPUT_DIR)
        
        if success:
            successful += 1
            status = f"{message}"
        else:
            failed += 1
            status = f"{message}"
        
        print(f"[{idx+1}/{len(df)}] {filename:45s} {status}")
        time.sleep(0.2) 
    
    # Summary
    elapsed = time.time() - start_time
    print("DOWNLOAD COMPLETE")
    print(f"Successful: {successful}/{len(df)}")
    print(f"Failed: {failed}/{len(df)}")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    
    
    if TEST_MODE and successful > 0:
        print("Test successful!")
        print(f"   Check the images in {OUTPUT_DIR} to verify quality")


if __name__ == "__main__":
    main()