#!/bin/bash

SAMPLE_RATE=16000

# fetch_clip(videoID, startTime, endTime)
  echo "Fetching $1 ($2,  $3), save in $4"

  outname="$4"
  $5 https://youtube.com/watch?v=$1 \
    --quiet --extract-audio --audio-format wav \
    --output $outname
  if [ $? -eq 0 ]; then
    # If we don't pipe `yes`, ffmpeg seems to steal a
    # character from stdin. I have no idea why.
    yes | ffmpeg -loglevel quiet -i $outname -ar $SAMPLE_RATE \
      -ss "$2" -t "$3" "${outname}_out.wav"
    mv "${outname}_out.wav" "$outname"
  fi

