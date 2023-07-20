# make sure directory exists
./x64/Release/buddhabrot.exe -w 100 -h 100 -s 50000 -i 1000 -et 5 -o output/test1_ -steps 24 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 90

# use ffmpeg to compile the gif that should look like the tour demo
yes y | ./ffmpeg.exe -f image2 -framerate 12 -i output/test1_%d.png output/test1.gif