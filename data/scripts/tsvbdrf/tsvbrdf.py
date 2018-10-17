import sys
import os
import optparse
import subprocess
import multiprocessing

parser = optparse.OptionParser()
parser.add_option('-m', '--mitsuba', help='the mitsuba executable', default='../../../build/binaries/Release/mitsuba.exe')
parser.add_option('-f', '--frames', help='number of frames', default=50)
parser.add_option('-i', '--input', help='input scene name (xml file)')
parser.add_option('-o', '--output', help='output directory')
parser.add_option('-M', '--material', help='material filepath')
(options, args) = parser.parse_args()

if(options.input == None):
    print("Need input values.\n")
    parser.print_help()
    sys.exit(1)
if(options.output == None):
    print("Need output values.\n")
    parser.print_help()
    sys.exit(1)
if not(0 < int(options.frames) < 300):
    print("Invalid number of frames.\n")
    parser.print_help()
    sys.exit(1)

if(not os.path.exists(options.output)):
    os.makedirs(options.output)

time = 0
frames = int(options.frames)
framerate = 1 / float(frames - 1)
for i in range(0, frames):
    command = [options.mitsuba, options.input]
    command += ["-o", options.output + os.path.sep + str(i)]
    command += ["-p", str(multiprocessing.cpu_count() - 1)]
    command += ["-Dtime=" + str(time)]
    command += ["-Dmaterial=" + options.material]
    #command += ["-z"]
    print(command)
    subprocess.call(command)
    time += framerate
