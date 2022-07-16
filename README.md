# buddhabrot
Utitilty to generate density plot of mandelbrot set trajectories.

-----

|![](https://github.com/mallocc/buddhabrot/blob/main/tour_demo.gif)|
|:--:|
|`buddhabrot.exe -w 100 -h 100 -s 50000 -i 1000 -et 5 -o output/test1_ -steps 24 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 10 -a 60 -nc -a 89 -b 89 -nc -b 90 -a 90 -nc -a 90.5 -b 90.5 -steps 96 -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 48 -nc -b 0 -steps 12 -nc -steps 96 -nc -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629`|
|![](https://github.com/mallocc/buddhabrot/blob/main/nebula_demo.png)|
|`buddhabrot.exe -w 720 -h 720 -s 100000000 -ir 2000 -ig 200 -ib 20 -x0 -2 -y0 -1.5 -x1 1 -y1 1.5`|

-----

# Intro
This program aims to make it easy to generate buddhabrot images and compile animations. It has various options that can be changed to stylise your image. Transformations can be made to the render, such as translation and rotation. Animations can be made using the staging feature, where images are interpolated between key frames (stages).

# Dependancies
The only dependancy is the famous `stb_image_write.h` header, for creating the image files. Also, I've added my solution file for anyone that likes Visual Studio.

# Getting started
I've added a shell script `test.sh` that will attempt to generate the anination gif above, so to check everything is working, and to see an example usage.

# Usage/Options

We can added many options to the program to specify how we want the image to look:

 - `-w | --width` - the width of the image (px)
 - `-h | --height` - the height of the image (px)
 - `-i | --iterations` - max iteratons used for a greyscale image
 - `-ir | --iterations-red` - max iteratons used for the red channel on an RGB image
 - `-ig | --iterations-green` - max iteratons used for the green channel on an RGB image
 - `-ib | --iterations-blue` - max iteratons used for the blue channel on an RGB image
 - `-im | --iterations-min` - min iterations used
 - `--gamma` - gamma correction value for the colouring (used for 'sqrt colouring')
 - `--radius` - radius bounds used in the bailout condition
 - `-x0 | -re0 | --real0` - (STAGE) top LEFT corresponding complex coordinate value
 - `-y0 | -im0 | --imaginary0` - (STAGE) TOP left corresponding complex coordinate value
 - `-x1 | -re1 | --real1` - (STAGE) bottom RIGHT corresponding complex coordinate value
 - `-y1 | -im1 | --imaginary1` - (STAGE) BOTTOM right corresponding complex coordinate value
 - `-s | --samples` - samples used for each channel
 - `-o | --output` - output filename used (for animation, incremented integer will be a appended to the end, i.e. test0.png, test1.png, ...)
 - `--steps` - (STAGE) steps for the current stage in the animation
 - `-a | --alpha` - (STAGE) alpha rotation in degrees
 - `-b | --beta` - (STAGE) beta rotation in degrees
 - `-t | --theta` - (STAGE) theta rotation in degrees (offsets of alpha an beta equally)
 - `-et | --escape-trajectories` - for greyscale images, filters for iterations above this value (but below max iterations)
 - `-etr | --escape-trajectories-red` - for red channel, filters for iterations above this value (but below max red iterations)
 - `-etg | --escape-trajectories-green` - for green channel, filters for iterations above this value (but below max green iterations)
 - `-etb | --escape-trajectories-blue` - for blue channel, filters for iterations above this value (but below max blue iterations)
 - `--counter` - offsets the incremeted integer used when generating animation images (if you need to append to an existing set of images)
 - `-n | --next | --next-stage` - will push all of the STAGE options to the list, and reset values for the next stage in the options
 - `-nc | --next-cpy | --next-stage-copy` - will push all of the STAGE options to the list, but will keep the values from the last stage (useful for appending stages in the options)
 
 # Algorithm
 
 todo
 
 
