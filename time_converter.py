def convert_times(input_string):
    """
    Convert decimal time values to formatted time strings.
    Handles both regular times (MM.SS) and hour format (H.MM.SS)
    
    Args:
        input_string (str): String containing decimal times separated by spaces
            Can handle optional forward slashes and both time formats
    
    Returns:
        list: List of formatted time strings in "MM:SS" or "HH:MM:SS" format
    
    Examples:
        >>> convert_times("12.3 1.00.16")
        ['12:30', '1:00:16']
    """
    # Remove any forward slashes and split by whitespace
    times = input_string.replace('/', '').split()
    
    formatted_times = []
    
    for time in times:
        parts = time.split('.')
        
        if len(parts) == 3:  # Hour format (e.g., 1.00.16)
            hours, minutes, seconds = parts
            formatted_time = f"{hours}:{minutes.zfill(2)}:{seconds.zfill(2)}"
        
        elif len(parts) == 2:  # Regular format (e.g., 12.3 or 5.22)
            minutes, decimal = parts
            
            # Handle single digit decimals (e.g., 12.3 -> 12:30)
            if len(decimal) == 1:
                seconds = str(int(decimal) * 10).zfill(2)
            # Handle regular seconds notation (e.g., 14.05 -> 14:05)
            else:
                seconds = decimal.zfill(2)
            
            formatted_time = f"{minutes.zfill(2)}:{seconds}"
        
        else:  # Just minutes (e.g., 16)
            formatted_time = f"{parts[0].zfill(2)}:00"
        
        formatted_times.append(formatted_time)
    
    return formatted_times

# Test cases
test_inputs = [
"			8.2	10.1	11.45	13.2	14.53	16.27	18.01	19.36	21.1	22.44	24.16 / 24.39	26.12	27.46	29.2	30.54	32.28	34.02	35.36	37.1	38.43	40.17	41.5	43.35	45.13	46.47	48.2	49.54	51.27	53.21	55.29	57.22	59.12	1.01.21	1.02.55	1.04.29	1.06.02	1.07.35	1.09.08	1.10.41	1.12.14	1.13.47	1.15.21	1.16.54	1.18.27	1.19.59	1.21.32	1.23.05	1.24.37	1.26.10	1.27.43	1.29.15	1.30.49	1.32.24	1.34.04 / 1.34.27	1.35.58	1.37.29	1.39.01	1.40.32	1.43.17																		"

]
# Run tests
for test in test_inputs:
    print(f"Input: {test}")
    result = convert_times(test)
    print("Output:")
    print(result)
    print(f"Number of times converted: {len(result)}")