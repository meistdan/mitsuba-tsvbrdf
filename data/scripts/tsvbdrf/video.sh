#!/bin/bash

AVISYNTH="/cygdrive/c/Program Files (x86)/VirtualDub/vdub.exe"

"$AVISYNTH" /c /s submission.vcf /p clip.avs clip.avi /r /x
