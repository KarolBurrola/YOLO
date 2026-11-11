import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# Mapping dictionaries for character conversion
# Expanded character conversion dictionaries
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '8',
                    'Z': '2',
                    'T': '7'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B',
                    '2': 'Z',
                    '7': 'T'}

# Recognized format masks ('L' for letter, 'D' for digit)
# 1. Sonora / México particular (e.g. WYZ1234)
# 2. Sonora / México SCT nuevo (e.g. WYZ123A)
# 3. Sonora / México clásico (e.g. WYZ123)
# 4. UK / Europa demo original (e.g. CE61VYL)
SUPPORTED_MASKS = [
    ['L', 'L', 'L', 'D', 'D', 'D', 'D'],  # Sonora / México 7 (LLLDDDD)
    ['L', 'L', 'L', 'D', 'D', 'D', 'L'],  # Sonora SCT nuevo (LLLDDDL)
    ['L', 'L', 'L', 'D', 'D', 'D'],       # Sonora clásico 6 (LLLDDD)
    ['L', 'L', 'D', 'D', 'L', 'L', 'L'],  # UK original (LLDDLLL)
]

BLACKLIST_WORDS = {'SONORA', 'MEXICO', 'TRANSPORTE', 'PRIVADO', 'GOBIERNO', 'ESTADO', 'FRONTERA'}


def check_and_format_mask(text, mask):
    """
    Checks if text matches mask length and character types, correcting ambiguous characters.
    Tracks number of character conversions to prevent converting arbitrary text/numbers.
    """
    if len(text) != len(mask):
        return None, 999

    formatted = []
    conversions = 0
    digits = set(string.digits)
    letters = set(string.ascii_uppercase)

    for char, required_type in zip(text, mask):
        if required_type == 'L':
            if char in letters:
                formatted.append(char)
            elif char in dict_int_to_char:
                formatted.append(dict_int_to_char[char])
                conversions += 1
            else:
                return None, 999
        elif required_type == 'D':
            if char in digits:
                formatted.append(char)
            elif char in dict_char_to_int:
                formatted.append(dict_char_to_int[char])
                conversions += 1
            else:
                return None, 999
        else:
            return None, 999

    # Maximum 2 ambiguous conversions allowed to avoid hallucinating false positives
    if conversions > 2:
        return None, 999

    return ''.join(formatted), conversions


def format_license(text):
    """
    Format and validate license plate against Sonora, Mexican, and UK patterns.
    Picks the pattern requiring the minimum number of character conversions.
    """
    clean_text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

    if clean_text in BLACKLIST_WORDS:
        return None

    best_formatted = None
    min_conversions = 999

    for mask in SUPPORTED_MASKS:
        formatted, conv = check_and_format_mask(clean_text, mask)
        if formatted is not None and conv < min_conversions:
            best_formatted = formatted
            min_conversions = conv
            if min_conversions == 0:
                break

    return best_formatted


def license_complies_format(text):
    """
    Check if the license plate text complies with any supported format.
    """
    return format_license(text) is not None


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from cropped image, with fallback to central band crop
    to discard state banners like 'SONORA' or 'MEXICO'.
    """
    if license_plate_crop is None or getattr(license_plate_crop, 'size', 0) == 0:
        return None, None

    # Step 1: Read full crop
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        clean_text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

        if clean_text in BLACKLIST_WORDS:
            continue

        formatted_plate = format_license(clean_text)
        if formatted_plate is not None:
            return formatted_plate, score

    # Step 2: Fallback for Mexican plates - crop out top 18% and bottom 18% (state banners)
    h, w = license_plate_crop.shape[:2]
    if h > 20 and w > 40:
        central_crop = license_plate_crop[int(h * 0.18):int(h * 0.82), :]
        detections_central = reader.readtext(central_crop)

        for detection in detections_central:
            bbox, text, score = detection
            clean_text = text.upper().replace(' ', '').replace('-', '').replace('.', '')

            if clean_text in BLACKLIST_WORDS:
                continue

            formatted_plate = format_license(clean_text)
            if formatted_plate is not None:
                return formatted_plate, score

    return None, None


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
