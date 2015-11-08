# Quick script for posterity for getting the number of frames for each file
# Usage:
# (1) Copy this file to the location of orig directory (e.g., 
# 	"/cygdrive/C/.../ASLrecog_private/data/orig/")
# (2) Make getFrames.sh executable
#	chmod +x getFrames.sh
# (2) Execute in that directory and copy results to a file for Python's delight:
#   ./getFrames.sh > frameCounts.txt
#
# Thanks to LordNeckbeard on StackOverflow for getting frame counts:
#    http://stackoverflow.com/questions/2017843/fetch-frame-count-with-ffmpeg

mp4files=( */*/*/*.mp4 )
for d in "${mp4files[@]}"; do
  printf '%s\n' "$d"
  ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$d"
done