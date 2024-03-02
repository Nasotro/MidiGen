import mido
import time

# Create a new MIDI file
midi_file = mido.MidiFile()

# Create a new MIDI track
track = mido.MidiTrack()
midi_file.tracks.append(track)

# Define the MIDI note numbers for the notes you want to create (e.g., C4, D4, E4, etc.)
note_numbers = [60, 62, 64, 65]  # You can add more notes as needed

# Define the start times for each note (in seconds)
start_times = [mido.tick2second(tick, ) for tick in range(len(note_numbers))]

# Set the tempo (in microseconds per quarter note)
tempo = mido.MetaMessage('set_tempo', tempo=int(1e6 / 120))  # 120 BPM
track.append(tempo)

# Create notes with 1-second duration
for note_number, start_time in zip(note_numbers, start_times):
    on_message = mido.Message('note_on', note=note_number, velocity=64, time=int(start_time * 1000))
    off_message = mido.Message('note_off', note=note_number, velocity=64, time=int((start_time + 1) * 1000))
    track.append(on_message)
    track.append(off_message)

# Save the MIDI file
midi_file.save('one_second_notes_mido.mid')
