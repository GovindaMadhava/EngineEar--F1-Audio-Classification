from pydub import AudioSegment
import os

def time_to_milliseconds(time_str):
    parts = time_str.split(':')
    if len(parts) == 3:  # Format: "hours:minutes:seconds"
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return int((hours * 3600 + minutes * 60 + seconds) * 1000)
    elif len(parts) == 2:  # Format: "minutes:seconds"
        minutes = int(parts[0])
        seconds = float(parts[1])
        return int((minutes * 60 + seconds) * 1000)
    elif len(parts) == 1:  # Format: "seconds" only
        seconds = float(parts[0])
        return int(seconds * 1000)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def split_audio(file_path, times_list):
    # Extract the inner list if times_list is nested
    if len(times_list) == 1 and isinstance(times_list[0], list):
        times = times_list[0]
    else:
        times = times_list
    
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Create a directory for chunks if it doesn't exist
    parent_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).replace('.wav', '')
    output_dir = os.path.join(parent_dir, f"{file_name}_LAPS")
    os.makedirs(output_dir, exist_ok=True)

    # Convert times array to milliseconds
    times_ms = [time_to_milliseconds(time) for time in times]
    
    # Add start and end times
    times_ms = [0] + times_ms + [len(audio)]
    
    # Split the audio into chunks
    for i in range(len(times_ms) - 1):
        start_time = times_ms[i]
        end_time = times_ms[i + 1]
        chunk = audio[start_time:end_time]
        
        # Name the output chunk
        chunk_name = f"{file_name}_lap{i}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)
        
        # Export the chunk
        chunk.export(chunk_path, format="wav")
        print(f"Exported: {chunk_path}")

# Example usage
file_path = "/Volumes/Sandhya TB2/F1_RACE_FOOTAGE_CLEANED/MIAMI/WilliamsMercedes_AlexanderAlbon23_MiamiIntlAutodrome^USA.wav"
times = [
['08:20', '10:10', '11:45', '13:20', '14:53', '16:27', '18:01', '19:36', '21:10', '22:44', '24:16', '24:39', '26:12', '27:46', '29:20', '30:54', '32:28', '34:02', '35:36', '37:10', '38:43', '40:17', '41:50', '43:35', '45:13', '46:47', '48:20', '49:54', '51:27', '53:21', '55:29', '57:22', '59:12', '1:01:21', '1:02:55', '1:04:29', '1:06:02', '1:07:35', '1:09:08', '1:10:41', '1:12:14', '1:13:47', '1:15:21', '1:16:54', '1:18:27', '1:19:59', '1:21:32', '1:23:05', '1:24:37', '1:26:10', '1:27:43', '1:29:15', '1:30:49', '1:32:24', '1:34:04', '1:34:27', '1:35:58', '1:37:29', '1:39:01', '1:40:32', '1:43:17']

]
split_audio(file_path, times)