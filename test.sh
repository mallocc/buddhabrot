# make sure directory exists
./x64/Release/buddhabrot.exe -w 100 -h 100 -s 50000 -i 1000 -et 5 -o output/test1_ -steps 24 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 10 -a 60 -nc -a 89 -b 89 -nc -b 90 -a 90 -nc -a 90.5 -b 90.5 -steps 96 -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 48 -nc -b 0 -steps 12 -nc -steps 96 -nc -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629

# use ffmpeg to compile the gif that should look like the tour demo
./ffmpeg.exe -f image2 -framerate 12 -i output/test1_%d.png output/test1.gif
