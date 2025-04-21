import os
import csv

def generate_f1_dataset_csv(parent_dir, output_csv):
    """
    Generate a CSV file containing F1 lap data information from audio files.
    
    Args:
        parent_dir (str): Path to the parent directory containing F1 lap data
        output_csv (str): Path where the output CSV file will be saved
    """
    # List to store all data entries
    data = []
    serial_number = 1
    
    # Walk through the parent directory
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        
        # Skip if not a directory or if it starts with '.'
        if not os.path.isdir(folder_path) or folder_name.startswith('.'):
            continue
            
        # Process each audio file in the folder
        for file_name in os.listdir(folder_path):
            # Skip hidden files and non-wav files
            if file_name.startswith('.') or not file_name.endswith('.wav'):
                continue
                
            # Get full file path
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # Split by underscore to get components
                # Example: AlpineRenault_PierreGasly10_AutodromoNazionaleMonza^Italy_lap20.wav
                parts = file_name.split('_')
                
                if len(parts) >= 3:  # Ensure we have at least team, driver, and track
                    team_name = parts[0]
                    driver_name = parts[1]
                    # Track name is the third part (remove .wav and anything after last underscore)
                    track_name = parts[2].split('_')[0]
                    
                    # Create data entry
                    entry = [
                        serial_number,
                        team_name,
                        driver_name,
                        track_name,
                        file_path
                    ]
                    
                    data.append(entry)
                    serial_number += 1
                
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
                continue
    
    # Write to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'Serial Number',
            'Team Name',
            'Driver Name',
            'Track Name',
            'File Path'
        ])
        
        # Write data rows
        writer.writerows(data)
    
    print(f"Successfully generated CSV file at: {output_csv}")
    print(f"Total entries processed: {serial_number - 1}")

# Example usage
if __name__ == "__main__":
    F1_LAP_DATA_DIR = "/Volumes/Sandhya TB2/F1_LAP_DATA"
    OUTPUT_CSV = "/Users/govindamadhavabs/F1_12lap_dataset.csv"
    
    generate_f1_dataset_csv(F1_LAP_DATA_DIR, OUTPUT_CSV)
