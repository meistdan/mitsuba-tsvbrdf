
COMPILE=../../../compile_video.exe
AVISYNTH="c:/Program Files (x86)/VirtualDub/vdub.exe"

.SUFFIXES: .avi .avs

DEFAULT:	clip.avi 

.avs.avi:
	@echo "Converting $< to $@ ..."
	$(AVISYNTH) /c /s "submission.vcf" /p "$<" "$@" /r /x
