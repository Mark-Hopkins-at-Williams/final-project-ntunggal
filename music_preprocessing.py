import music21 
from music21 import converter, note, pitch 
from music21.stream import Part, Measure 
import glob 
import torch.nn.functional as F
import os


MIDI_FILE_PATH = "./midi_songs/train" ## this is the directory where audio music files are stored. 
PROCESSED_DATASET_DIRECTORY = "__data_processing__" ## this saves the music, for debugging purposes. 
SAVING_DATASET_DIRECTORY = "dataset_for_all_midi"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4,
]
SEQUENCE_LENGTH = 64


############################################################
############# DO NOT CHANGE ANYTHING BELOW  ################ 
############################################################


def get_all_music_from_dataset(directory = MIDI_FILE_PATH):
    """
    Loads all music files from the specified directory.
    
    Parameters:
        directory (str): Path to the directory containing MIDI files.
    
    Output:
        list: A list of music21 stream objects.
    """
    list_of_music = [] 
    for file in glob.glob(os.path.join(directory, "*.krn")):
        music = converter.parse(file) 
        list_of_music.append(music) 
    return list_of_music


def acceptable_durations(music, acceptable_durations = ACCEPTABLE_DURATIONS):
    """
    Checks if all notes and rests in the music have acceptable durations.
    
    Parameters:
        music (music21.stream.Score): The music to check.
        acceptable_durations (list): List of acceptable durations.
    
    Output:
        bool: True if all durations are acceptable, False otherwise.
    """
    for note in music.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(music):
    """
    Transposes music to C major or A minor.
    
    Parameters:
        music (music21.stream.Score): The music to transpose.
    
    Output:
        music21.stream.Score: The transposed music.
    """
    parts = music.getElementsByClass(Part)
    measurePart0 = parts[0].getElementsByClass(Measure)
    
    music_key = measurePart0[0][4] ## get the key singnature, since the key is always in the first measure. 
    if not isinstance(music_key, music21.key.Key):
        music_key = music.analyze("key")
   
    interval = None 
    if music_key.mode == "major":
        interval = music21.interval.Interval(music_key.tonic, pitch.Pitch("C"))
    elif music_key.mode == "major": 
        interval = music21.interval.Interval(music_key.tonic, pitch.Pitch("A"))
    music_transposed = music.transpose(interval) 
    return music_transposed


def music_encoder(music, time_step = 0.25):
    """
    Encodes music into a string format suitable for training.
    
    Parameters:
        music (music21.stream.Score): The music to encode.
        time_step (float): The time step for encoding.
    
    Returns:
        str: The encoded music as a string.
    """
    list_of_encoded_music = [] 
    for event in music.flat.notesAndRests: 
        symbol = None 
        if isinstance(event, note.Note):  
            symbol = event.pitch.midi
        elif isinstance(event, note.Rest):  
            symbol = "r"  # use 'r' to denote rest in music
        steps = int(event.duration.quarterLength / time_step) 
        for step in range(steps): 
            if step == 0: 
                list_of_encoded_music.append(symbol) 
            else: 
                list_of_encoded_music.append('_')
    encoded_music = " ".join(map(str, list_of_encoded_music))  
    return encoded_music


def preprocess_midi_files():
    """
    Preprocesses MIDI files: loads, filters, transposes, and encodes them.
    
    Returns:
        dict: A dictionary mapping music index to its encoded string.
    """
    print(f"PRE-PROCESSING MUSIC FILES...")
    list_of_music = get_all_music_from_dataset()
    print(f"Loaded a total of {len(list_of_music)} music")
    preprocessed_output = {}
    for music_index, music in enumerate(list_of_music):
        if not acceptable_durations(music):
            continue
        transpoed_music = transpose(music) 
        encoded_music = music_encoder(transpoed_music) 
        path_to_save_encoded_music = os.path.join(PROCESSED_DATASET_DIRECTORY, str(music_index)) 
        os.makedirs(os.path.dirname(path_to_save_encoded_music), exist_ok=True)
        with open(path_to_save_encoded_music, 'w') as f: 
            f.write(encoded_music)
        preprocessed_output[music_index] = encoded_music
    return preprocessed_output 