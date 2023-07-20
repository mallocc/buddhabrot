# make sure directory exists

args=()

#args+="-vbx cr -vby ci -vbz zi "
#args+="-steps 100 "
#args+="-x0 -1.7964999 -y0 -0.0268000 -x1 -1.7250449 -y1 0.026830 -nc "  #"-x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 0 -t 0"
#args+="-a 360 -b 360 -nc "
#args+="-x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 0 -t 0 -b -45 -nc "
#args+="-a 45 -b 90 -nc "
#args+="-a 45 -b -45 -nc "
#args+="-a -90 -b 0 -nc "
#args+="-nc -b 360 "
args+="-x0 -1.7964999 -y0 -0.0268000 -x1 -1.7250449 -y1 0.026830 "
#args+="-steps 24 -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -nc -a 30 -b 60"
#args+="-steps 100 -nc -a 180 -b 180 "
#args+="-a 135 -b 90 -nc "
#args+="-a 135 -b 135 -nc "
#args+="-b 90 -a 90 -nc "
#args+="-a 90.5 -b 90.5 -steps 96 -nc "
#args+="-x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 48 -nc "
#args+="-b 0 -steps 12 -nc "
#args+="-steps 96 -nc "
#args+="-x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 "

#args+="-a 45 -b 45 "

#echo ${args[*]}

#./x64/Release/buddhabrot.exe -w 500 -h 500 -s 1000000 -i 1000 -et 20 -o output/test2_ ${args[*]}

#./x64/Release/buddhabrot.exe -crop-samples-enable -random-disable -gamma 2.5 -w 1280 -h 1280 -s 8 -ir 5000 -ig 1000 -ib 100 -et 0 -a 30 -b -60 -o images/test10 ${args[*]}

./x64/Release/buddhabrot.exe -crop-samples-enable -random-enable -gamma 2.5 -w 512 -h 512 -s 262144 -ir 5000 -ig 1000 -ib 100 -et 0 -o images/test14 ${args[*]}

#./x64/Release/buddhabrot.exe -bezier-enable -crop-samples-enable -random-disable -gamma 3 -w 256 -h 256 -s 1 -i 1000 -im 50 -o output/test13_ ${args[*]}
#yes y | ./ffmpeg.exe -f image2 -framerate 12 -i output/test13_%d.png output/test13.gif
