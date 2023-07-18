#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <random>
#include <stack>
#include <functional>
#include <filesystem>
#include <map>
#include <chrono>

#include "Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Delicious
#define PI 3.1415926

/**
 *
 * Mersenne Twister random number generator.
 *
 */
double randf(const double range = 1.0, bool sign = false)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	return dis(gen) * range * (1.0 + (double)sign) - range * sign;
}

template<class T>
class Timer {
public:
	Timer() : totalDuration(0), numSamples(0) {}

	void start() {
		startTime = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<T>(endTime - startTime).count();

		totalDuration.fetch_add(duration);
		numSamples.fetch_add(1);
	}

	double getAverageTime() const {
		if (numSamples == 0) {
			return 0.0;
		}
		return static_cast<double>(totalDuration) / numSamples;
	}

	double getRemainingTime(int totalIterations, int completedIterations) const {
		int remainingIterations = totalIterations - completedIterations;
		if (remainingIterations <= 0) {
			return 0.0;
		}
		double averageTime = getAverageTime();
		return averageTime * remainingIterations;
	}

private:
	std::chrono::high_resolution_clock::time_point startTime;
	std::atomic<long long> totalDuration;
	std::atomic<int> numSamples;
};

template<class T>
std::string getTimeUnit() {
	if (std::is_same<T, std::chrono::nanoseconds>::value) {
		return "ns";
	}
	else if (std::is_same<T, std::chrono::microseconds>::value) {
		return "us";
	}
	else if (std::is_same<T, std::chrono::milliseconds>::value) {
		return "ms";
	}
	else if (std::is_same<T, std::chrono::seconds>::value) {
		return "s";
	}
	else if (std::is_same<T, std::chrono::minutes>::value) {
		return "min";
	}
	else if (std::is_same<T, std::chrono::hours>::value) {
		return "hr";
	}
	else {
		return "Unknown";
	}
}

template<class T>
double convertToSeconds(double duration) {
	if (std::is_same<T, std::chrono::nanoseconds>::value) {
		return duration / 1e9;
	}
	else if (std::is_same<T, std::chrono::microseconds>::value) {
		return duration / 1e6;
	}
	else if (std::is_same<T, std::chrono::milliseconds>::value) {
		return duration / 1e3;
	}
	else if (std::is_same<T, std::chrono::seconds>::value) {
		return duration;
	}
	else if (std::is_same<T, std::chrono::minutes>::value) {
		return duration * 60;
	}
	else if (std::is_same<T, std::chrono::hours>::value) {
		return duration * 3600;
	}
	else {
		return 0.0;
	}
}

/**
 * Very basic complex struc with operations.
 */
struct Complex
{
	double re, im;

	Complex() : re(0.0), im(0.0) {}
	Complex(double re, double im) : re(re), im(im) {}

	Complex add(const Complex& a) { return Complex(re + a.re, im + a.im); }
	Complex operator+(const Complex& a) { return add(a); }
	Complex sub(const Complex& a) { return Complex(re - a.re, im - a.im); }
	Complex operator-(const Complex& a) { return sub(a); }
	Complex mult(const Complex& a) {
		return Complex(re * a.re - im * a.im,
			re * a.im + im * a.re);
	}
	Complex operator*(const Complex& a) { return mult(a); }

	Complex operator/(double a) { return Complex(re / a, im / a); }
	Complex operator*(double a) { return Complex(re * a, im * a); }

	double mod2() { return re * re + im * im; }

	operator std::string() {
		return "(" + std::to_string(re) + ", "
			+ std::to_string(im) + "i)";
	}
};

/**
 * Main buddhabrot rendering structure.
 */
struct BuddhabrotRenderer
{
	/**
	 * Stage struct to hold info for animation key frame.
	 */
	struct Stage
	{
		Complex v0 = Complex(-2, -1.5);
		Complex v1 = Complex(1, 1.5);
		double alpha = 0;
		double beta = 0;
		double theta = 0;
		int steps = 1;
		double gamma = 2;
		bool bezier = false;
	};

	BuddhabrotRenderer() {}

	uint8_t* pixelData = nullptr;
	int* buddhaData = nullptr;

	std::string filename = "";

	int width = 0;
	int height = 0;
	int components = 1;
	float samples = 0;
	int iterations = 0;
	int iterationsR = 0;
	int iterationsG = 0;
	int iterationsB = 0;
	double radius = 4.0;
	bool isAnti = false;
	int escapeThreshold = 0;
	int escapeThresholdR = 0;
	int escapeThresholdG = 0;
	int escapeThresholdB = 0;
	int iterationsMin = 0;

	int superSampleFactor = 1;

	bool cropSamples = true;

	bool useRandomGeneration = false;

	int volumeAX = 0;
	int volumeAY = 1;
	int volumeAZ = 2;

	int volumeBX = 0;
	int volumeBY = 1;
	int volumeBZ = 2;

	Complex zr;
	Complex cr;

	int counter = 0;

	std::vector<Stage> stages;

	Timer<std::chrono::milliseconds> timer;

	float bmTime = 0.0f;

	float benchmark()
	{
		const int numIterations = 1000000; // Adjust the number of iterations as desired

		Timer<std::chrono::nanoseconds> timer;
		timer.start();
		double randomValue = 0.0f;
		for (int i = 0; i < numIterations; ++i)
		{
			randomValue *= randf(1.0, true);
			// Do something with the random value if necessary
		}
		timer.stop();
		return 7.0f * timer.getAverageTime() / numIterations;
	}

	// This should initialise the data arrays and another 
	// members that's value are derived from other.
	void init()
	{
		components = (iterationsR > 0
			|| iterationsG > 0
			|| iterationsB > 0) ? 3 : 1;

		buddhaData = new int[width * height * superSampleFactor * superSampleFactor];
		pixelData = new uint8_t[width * height * components];
	}

	// Zeros all of the buddhaData array
	void clearBuddhaData()
	{
		for (int i = 0; i < width * height * superSampleFactor * superSampleFactor; ++i)
			buddhaData[i] = 0;
	}

	// Zeros both buddhaData and pixelData arrays
	void clearAll()
	{
		clearBuddhaData();
		for (int i = 0; i < width * height * components; ++i)
			pixelData[i] = 0;
	}

	// Processes a single frame with the provided properties
	void processFrame(const Complex& v0, const Complex& v1,
		const Complex& zr, const Complex& cr,
		const double alphaL, const double betaL, const double thetaL,
		const double gamma,
		const int step)
	{
		bool componentOverride = false;

		if (iterationsR > 0)
		{
			LOG("Processing red channel... ");
			process(buddhaData,
				width * superSampleFactor, height * superSampleFactor, samples, iterationsR, radius,
				v0, v1, zr, cr,
				alphaL, betaL, thetaL,
				false, escapeThresholdR, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				gamma, 0);
			clearBuddhaData();
			componentOverride = true;
		}
		if (iterationsG > 0)
		{
			LOG("Processing green channel... ");
			process(buddhaData, width * superSampleFactor, height * superSampleFactor, samples, iterationsG, radius,
				v0, v1, zr, cr,
				alphaL, betaL, thetaL,
				false, escapeThresholdG, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				gamma, 1);
			clearBuddhaData();
			componentOverride = true;
		}
		if (iterationsB > 0)
		{
			LOG("Processing blue channel... ");
			process(buddhaData, width * superSampleFactor, height * superSampleFactor, samples, iterationsB, radius,
				v0, v1, zr, cr,
				alphaL, betaL, thetaL,
				false, escapeThresholdB, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				gamma, 2);
			clearBuddhaData();
			componentOverride = true;
		}

		if (!componentOverride)
		{
			process(buddhaData, width * superSampleFactor, height * superSampleFactor, samples, iterations, radius,
				v0, v1, zr, cr,
				alphaL, betaL, thetaL,
				false, escapeThreshold, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				gamma);
		}

		writeToPNG(filename.empty() ? "" : filename + std::to_string(step),
			width, height, components, pixelData);
	}


	// Linear interpolation
	static double b1(double x0, double x1, double t)
	{
		return x0 + (x1 - x0) * t;
	}

	// Quadratic interpolation
	static double b2(double x0, double x1, double x2, double t)
	{
		return pow(1 - t, 2) * x0 + 2 * (1 - t) * t * x1 + t * t * x2;
	}

	// Cubic interpolation
	static double b3(double x0, double x1, double x2, double x3, double t)
	{
		return pow(1 - t, 3) * x0
			+ 3 * t * pow(1 - t, 2) * x1
			+ 3 * pow(t, 2) * (1 - t) * x2
			+ pow(t, 3) * x3;
	}

	// Runs the renderer with the options specified
	void run()
	{
		init();

		bmTime = benchmark();

		LOG(
			"Render details:" << std::endl
			<< "\t " << width << "x" << height << std::endl
			<< "\t samples = " << samples << std::endl
			<< "\t iterations = " << iterations << std::endl
			<< "\t iterations-red = " << iterationsR << std::endl
			<< "\t iterations-green = " << iterationsG << std::endl
			<< "\t iterations-blue = " << iterationsB << std::endl
			<< "\t color-components = " << components << std::endl
			<< "\t radius = " << radius << std::endl
			<< "\t x0 = " << stages[0].v0.re << " y0 = " << stages[0].v0.im << std::endl
			<< "\t x1 = " << stages[0].v1.re << " y1 = " << stages[0].v1.im << std::endl
			<< "\t alpha = " << stages[0].alpha << std::endl
			<< "\t beta = " << stages[0].beta << std::endl
			<< "\t theta = " << stages[0].theta << std::endl
			<< "\t escaping-trajectories = " << escapeThreshold << std::endl
			<< "\t stages = " << stages.size() << std::endl
			<< "\t bmTime = " << bmTime << std::endl
		);

		int totalSteps = 0;
		for (int stage = 0; stage < stages.size() - 1; ++stage)
			totalSteps += stages[stage].steps;

		if (stages.size() > 1)
		{
			int stepC = 0;
			for (int stage = 0; stage < stages.size() - 1; ++stage)
				for (int step = 0, steps = stages[stage].steps; step < steps; ++step, ++stepC)
				{
					LOG("Processing stage " << stage << " / " << stages.size() - 2 << ", step " << step << " / " << steps - 1 << "...");
					clearAll();

					double alphaL = stages[stage].alpha;
					double betaL = stages[stage].beta;
					double thetaL = stages[stage].theta;
					Complex v0 = stages[stage].v0;
					Complex v1 = stages[stage].v1;
					double gamma = stages[stage].gamma;

					if (stages.size() > 1)
					{
						double b = stages[stage].bezier ? b3(0, 0, 1, 1, step / (double)steps) : (step / (double)steps); // ease in out
						//double b = b3(0, 1, 1, 1, step / (double)steps); // ease in
						//double b = b3(0, 1, 0, 1, step / (double)steps); // ease out
						alphaL = (b * (stages[stage + 1].alpha - stages[stage].alpha) + stages[stage].alpha) / 180 * PI;
						betaL = (b * (stages[stage + 1].beta - stages[stage].beta) + stages[stage].beta) / 180 * PI;
						thetaL = (b * (stages[stage + 1].theta - stages[stage].theta) + stages[stage].theta) / 180 * PI;
						gamma = (b * (stages[stage + 1].gamma - stages[stage].gamma) + stages[stage].gamma);
						v0.re = (b * (stages[stage + 1].v0.re - stages[stage].v0.re) + stages[stage].v0.re);
						v0.im = (b * (stages[stage + 1].v0.im - stages[stage].v0.im) + stages[stage].v0.im);
						v1.re = (b * (stages[stage + 1].v1.re - stages[stage].v1.re) + stages[stage].v1.re);
						v1.im = (b * (stages[stage + 1].v1.im - stages[stage].v1.im) + stages[stage].v1.im);
					}

					timer.start();

					processFrame(v0, v1, zr, cr, alphaL, betaL, thetaL, gamma, stepC + counter);

					timer.stop();
					LOG("Average time: " << timer.getAverageTime() / 1000.0f << "s, estimated time left: " << (totalSteps - stepC) * timer.getAverageTime() / 1000.0f << "s");
				}
		}
		else if (!stages.empty())
		{
			clearAll();
			processFrame(stages[0].v0, stages[0].v1, zr, cr, stages[0].alpha / 180 * PI, stages[0].beta / 180 * PI, stages[0].theta / 180 * PI, stages[0].gamma, 0);
		}
	}

	// map dimension name to index
	std::map<std::string, int> dimensions = {
		{"zr", 0},
		{"zi", 1},
		{"cr", 2},
		{"ci", 3}
	};

	// This is the main buddhabrot algorithm in one function
	void process(
		int* data, int w, int h, float samples, int iter, int radius = 4.0,
		const Complex& minc = Complex(-2, -2),
		const Complex& maxc = Complex(2, 2),
		const Complex& zr = Complex(), const Complex& cr = Complex(),
		double alpha = 0, double beta = 0, double theta = 0, bool anti = false,
		int threshold = 0, int floorIter = 0, int threadCount = 0)
	{
		// pre commpute //

		// Rotation matrix coefficients
		double Axx, Axy, Axz, Ayy, Ayz, Azx, Azy, Azz;

		// Calculate rotation matrix coefficients
		double cosb = std::cos(alpha);
		double sinb = std::sin(alpha);

		double cosc = std::cos(beta);
		double sinc = std::sin(beta);

		Axx = cosb;
		Axy = sinb * sinc;
		Axz = sinb * cosc;

		Ayy = cosc;
		Ayz = -sinc;

		Azx = -sinb;
		Azy = cosb * sinc;
		Azz = cosb * cosc;

		// flag for if we are using the minimum threshold
		bool escapeColouring = threshold > 0;

		// find the size of the viewable complex plane
		Complex size = Complex(maxc) - minc;

		// for use in the loop for converting back to screen space
		double cw = w / (size.re);
		double ch = h / (size.im);

		// the center of the viewable complex plane
		Complex center = size / 2.0 + minc;

		std::vector<double> coords = { 0.0f, 0.0f, 0.0f, 0.0f };

		// find the OpenMP thread count
		if (threadCount == 0) {
#pragma omp parallel
			{
				threadCount = omp_get_num_threads();
			};
		}

		float incw = size.re / (float)(w * samples); // Width of each sample
		float inch = size.im / (float)(h * samples); // Height of each sample

		float incwF = 4.0f / (float)(w * samples); // Width of each sample 
		float inchF = 4.0f / (float)(h * samples); // Height of each sample

		int wsamples = w * samples;
		int hsamples = h * samples;

		auto trajectory = [&](int s, int ax = 0, int ay = 0) {
			// pre allocate potential iteration samples
			Complex* csamples = new Complex[iter];

			// initialise the mandelbrot components
			Complex c;
			if (!useRandomGeneration)
			{
				s = ay * wsamples + ax;
				if (cropSamples)
					c = Complex(ax * incw, ay * inch);
				else
					c = Complex(ax * incwF, ay * inchF);
				c = c + minc;
			}
			else if (cropSamples)
				c = { randf() * size.re + minc.re, randf() * size.im + minc.im };
			else
				c = { randf() * 4.0f - 2.0f, randf() * 4.0f - 2.0f };

			Complex z(c);

			// track i
			int i = 0;
			// we only care about trajetories that are less than max iterations 
			// in length and that they fall within the radius bounds
			for (; i < iter && z.mod2() < radius; ++i)
			{
				// translations through the shape (there are 4 axes)
				z = z + zr;
				c = c + cr;
				// apply the magic formula
				z = z * z + c;
				// store our sample complex position for later
				csamples[i] = z;
			}

			// if we want to rotate around a point, we must translate the point
			// to the origin first (we will do it for Z later, remember 4 axes)
			c = c - center;

			// filter for minimum iterations
			if (i >= floorIter)
				// flags to check between normal and anti brot
				if ((!anti && i < iter) || (anti && i == iter))
					// iterate through our valid iterations samples
					for (int j = 0; j < i; ++j)
					{
						// rotate around current center point //
						Complex& t = csamples[j];

						// if we want to rotate around a point, 
						// we must translate the point to the origin first
						t = t - center;

						coords[0] = t.re;
						coords[1] = t.im;
						coords[2] = c.re;
						coords[3] = c.im;

						// now apply the rotation matrix on t and c (these are
						// the points on the 4d volume)
						double x1 = coords[volumeAX];
						double y1 = coords[volumeAY];
						double z1 = coords[volumeAZ];
						double x2 = coords[volumeBX];
						double y2 = coords[volumeBY];
						double z2 = coords[volumeBZ];

						// Apply the rotation matrix on t
						double nx = x1 + (x2 - x1) * alpha;
						double ny = y1 + (y2 - y1) * beta;
						double nz = z1 + (z2 - z1) * theta;

						// Apply the rotation matrix coefficients
						double newX = Axx * nx + Axy * ny + Axz * nz;
						double newY = Ayy * ny + Ayz * nz;
						double newZ = Azx * nx + Azy * ny + Azz * nz;

						// Update the point coordinates
						t.re = newX;
						t.im = newY;

						// translate back to our point of interest
						t = t + center;

						// transform complex point into screen space point 
						// (screen space coord)
						int x = (t.re - minc.re) * cw;
						int y = (t.im - minc.im) * ch;
						// make sure it falls within in the screen buffer
						if (x >= 0 && x < w && y >= 0 && y < h)
							// incr each pixels components according to the 
							// colour thresholds
							data[(y * w + x)] += escapeColouring
							? j >= threshold
							: j < iter;
					}

			// clean up the potential iteration samples for this sample
			delete[] csamples;

			//if (!(s % (samples / 10)))
			//	LOG("Progress: " << std::setprecision(2) << (s / (double)samples) * 100.0 << "%...");

		};


		Timer<std::chrono::nanoseconds> ttimer;
		std::string timeUnit = getTimeUnit<std::chrono::nanoseconds>();
		const int printInterval = 1000000; // Adjust the interval as desired

		int sampleCount = 0; // Atomic counter to keep track of the sample number

		// leave a core for the rest of the OS
		if (useRandomGeneration)
#pragma omp parallel for num_threads(std::max(1, threadCount - 1))
			for (int s = 0; s < (int)samples; ++s)
			{
				trajectory(s);
#pragma omp atomic
				sampleCount++;

				if (sampleCount % printInterval == 0)
				{
					// Print the sample count
#pragma omp critical
					{
						std::cout << "Time left: " << convertToSeconds<std::chrono::nanoseconds>(bmTime * (samples - sampleCount)) << "s" << std::endl;
					}
				}
			}
		else
#pragma omp parallel for num_threads(std::max(1, threadCount - 1))
			for (int px = 0; px < wsamples; ++px)
				for (int py = 0; py < hsamples; ++py)
				{
					trajectory(0, px, py);
#pragma omp atomic
					sampleCount++;

					if (sampleCount % printInterval == 0)
					{
						// Print the sample count
#pragma omp critical
						{
							std::cout << "Time left: " << convertToSeconds<std::chrono::nanoseconds>(bmTime * (wsamples * hsamples - sampleCount)) << "s" << std::endl;
						}
					}
				}

		LOG("Progress: finished!");
	}

	// classic sqrt colouring using gamma correction
	// higher the gamma, the brighter it is
	static uint8_t sqrtColour(double x, double y, double gamma)
	{
		return pow(x / y, 1.0 / gamma) * UCHAR_MAX;
	}

	static double smoothstep(double x, double minVal, double maxVal)
	{
		// Ensure x is within the range [minVal, maxVal]
		x = std::clamp((x - minVal) / (maxVal - minVal), 0.0, 1.0);

		// Apply the smoothstep interpolation formula
		return x * x * (3 - 2 * x);
	}

	// Apply box blur to a 2D array
	static void boxBlur2D(int w, int h, int superSampleFactor, int* input, float* output, int radius)
	{
		int kernelSize = 2 * radius + 1;
		std::vector<int> kernel(kernelSize, 1);

		// Blur horizontally
		for (int y = 0; y < h * superSampleFactor; ++y)
		{
			for (int x = 0; x < w * superSampleFactor; ++x)
			{
				int sum = 0;
				for (int i = -radius; i <= radius; ++i)
				{
					int px = x + i;
					if (px < 0) px = 0;
					if (px >= w * superSampleFactor) px = w * superSampleFactor - 1;

					sum += input[y * w * superSampleFactor + px];
				}
				output[y * w * superSampleFactor + x] = sum / (float)kernelSize;
			}
		}

		// Blur vertically
		for (int y = 0; y < h * superSampleFactor; ++y)
		{
			for (int x = 0; x < w * superSampleFactor; ++x)
			{
				int sum = 0;
				for (int i = -radius; i <= radius; ++i)
				{
					int py = y + i;
					if (py < 0) py = 0;
					if (py >= h * superSampleFactor) py = h * superSampleFactor - 1;

					sum += input[py * w * superSampleFactor + x];
				}
				output[y * w * superSampleFactor + x] = sum / (float)kernelSize;
			}
		}
	}

	// Normalizes the buddhaData into pixelData with supersampling and box blur
	static void getPixelData(int w, int h, int c, int superSampleFactor, int* buddhaData, uint8_t* pixelData, double gamma = 2.0, int o = -1)
	{
		int supersampledW = w * superSampleFactor;
		int supersampledH = h * superSampleFactor;

		// Create a temporary buffer for supersampled data
		std::vector<float> supersampledBuddhaData(supersampledW * supersampledH);

		// Apply box blur to the supersampled buddhaData
		boxBlur2D(w, h, superSampleFactor, buddhaData, supersampledBuddhaData.data(), 1);

		// Calculate maximum value in the supersampled image
		// Find the minimum and maximum values
		float minValue = *std::min_element(supersampledBuddhaData.data(), supersampledBuddhaData.data() + supersampledBuddhaData.size());
		float maxValue = *std::max_element(supersampledBuddhaData.data(), supersampledBuddhaData.data() + supersampledBuddhaData.size());
		// Subtract the minimum value and divide by the range
		float range = maxValue - minValue;

		// Downscale the supersampled buddhaData to pixelData
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				double averagedValue = 0.0;
				for (int sy = 0; sy < superSampleFactor; ++sy)
				{
					for (int sx = 0; sx < superSampleFactor; ++sx)
					{
						int ssx = x * superSampleFactor + sx;
						int ssy = y * superSampleFactor + sy;
						averagedValue += supersampledBuddhaData[ssy * supersampledW + ssx];
					}
				}
				averagedValue /= (float)(superSampleFactor * superSampleFactor);

				for (int cc = 0; cc < c; ++cc)
				{
					if (o == -1 || o == cc)
					{
						pixelData[(y * w + x) * c + cc] = pow(smoothstep((averagedValue - minValue) / range, 0.0, 1.0f), 1.0 / gamma) * UCHAR_MAX;
					}
				}
			}
		}
	}

	// normalises the buddhaData into pixelData
	static void getPixelData2(int w, int h, int c, int superSampleFactor, int* buddhaData, uint8_t* pixelData, double gamma = 2.0, int o = -1)
	{
		double maxVal = 1;
		for (int i = 0; i < w * h; ++i)
			maxVal = std::max(maxVal, (double)buddhaData[i]);

		for (int i = 0; i < w * h; ++i)
			for (int cc = 0; cc < c; ++cc)
				if (o == -1 || o == cc)
					pixelData[i * c + cc] = sqrtColour(buddhaData[i], maxVal, gamma);
	}

	static bool createDirectories(const std::string& filepath) {
		std::filesystem::path path(filepath);

		// Extract the directory path
		std::filesystem::path directory = path.parent_path();

		// Create directories recursively
		try {
			std::filesystem::create_directories(directory);
		}
		catch (const std::filesystem::filesystem_error& e) {
			std::cerr << "Error creating directories: " << e.what() << std::endl;
			return false;
		}

		return true;
	}

	// writes pixelData out to a PNG using stb_image_write.h
	static void writeToPNG(const std::string& filename, int w, int h, int c, uint8_t* data)
	{
		LOG("Writing out render to PNG image...");
		auto time = currentISO8601TimeUTC();
		std::replace(time.begin(),
			time.end(),
			':',
			'_');
		std::stringstream ss;
		if (filename.empty())
			ss << "img_" << time << ".png";
		else
			ss << filename << ".png";
		createDirectories(ss.str().c_str());
		stbi_write_png(ss.str().c_str(), w, h, c, data, w * c);
	}

};

int main(int argc, char* argv[])
{
	LOG("Program entry");

	std::stack<std::string> args;
	for (int a = argc - 1; a >= 1; --a)
		args.push(argv[a]);

	BuddhabrotRenderer bb;

	bool success = true;

	BuddhabrotRenderer::Stage stage;

	// read argumetions and populate options
	while (!args.empty() && success)
	{
		std::string arg(args.top());
		args.pop();

		if (!arg.empty())
		{
			// a flag!
			if (arg[0] == '-')
			{
				std::string option(arg.substr(1));

				if (!option.empty())
				{
					if (option[0] == '-')
						option = option.substr(1);

					auto checkAndSet = [&](std::function<void(const std::string&)> callback)
					{
						if (!args.empty())
						{
							callback(args.top());
							args.pop();
						}
						else
						{
							LOG("No option value supplied: " << arg);
							success = false;
						}
					};

					auto checkAndSetAndReturn = [&](std::function<int(const std::string&)> callback)
					{
						if (!args.empty())
						{
							success = callback(args.top());
							LOG("Invalid option value supplied: " << args.top());
							args.pop();
						}
						else
						{
							LOG("No option value supplied: " << arg);
							success = false;
						}
					};

					auto checkSetVolumeVals = [&](int& val)
					{
						checkAndSetAndReturn([&](const std::string& in) -> bool
							{
								if (!bb.dimensions.contains(in))
									return false;
								val = bb.dimensions[in];
								return true;
							});
					};

					if (option == "w" || option == "width")
						checkAndSet([&](const std::string& in) { bb.width = std::stoi(in); });
					else if (option == "h" || option == "height")
						checkAndSet([&](const std::string& in) { bb.height = std::stoi(in); });
					else if (option == "vax" || option == "volume-a-x")
						checkSetVolumeVals(bb.volumeAX);
					else if (option == "vay" || option == "volume-a-y")
						checkSetVolumeVals(bb.volumeAY);
					else if (option == "vaz" || option == "volume-a-z")
						checkSetVolumeVals(bb.volumeAZ);
					else if (option == "vbx" || option == "volume-b-x")
						checkSetVolumeVals(bb.volumeBX);
					else if (option == "vby" || option == "volume-b-y")
						checkSetVolumeVals(bb.volumeBY);
					else if (option == "vbz" || option == "volume-b-z")
						checkSetVolumeVals(bb.volumeBZ);
					else if (option == "super-sample-factor")
						checkAndSet([&](const std::string& in) { bb.superSampleFactor = std::stoi(in); });
					else if (option == "i" || option == "iterations")
						checkAndSet([&](const std::string& in) { bb.iterations = std::stoi(in); });
					else if (option == "ir" || option == "iterations-red")
						checkAndSet([&](const std::string& in) { bb.iterationsR = std::stoi(in); });
					else if (option == "ig" || option == "iterations-green")
						checkAndSet([&](const std::string& in) { bb.iterationsG = std::stoi(in); });
					else if (option == "ib" || option == "iterations-blue")
						checkAndSet([&](const std::string& in) { bb.iterationsB = std::stoi(in); });
					else if (option == "im" || option == "iterations-min")
						checkAndSet([&](const std::string& in) { bb.iterationsMin = std::stoi(in); });
					else if (option == "gamma")
						checkAndSet([&](const std::string& in) { stage.gamma = std::stof(in); });
					else if (option == "radius")
						checkAndSet([&](const std::string& in) { bb.radius = std::stof(in); });
					else if (option == "re0" || option == "x0" || option == "real0")
						checkAndSet([&](const std::string& in) { stage.v0.re = std::stof(in); });
					else if (option == "im0" || option == "y0" || option == "imaginary0")
						checkAndSet([&](const std::string& in) { stage.v0.im = std::stof(in); });
					else if (option == "re1" || option == "x1" || option == "real1")
						checkAndSet([&](const std::string& in) { stage.v1.re = std::stof(in); });
					else if (option == "im1" || option == "y1" || option == "imaginary1")
						checkAndSet([&](const std::string& in) { stage.v1.im = std::stof(in); });
					else if (option == "s" || option == "samples")
						checkAndSet([&](const std::string& in) { bb.samples = std::stof(in); });
					else if (option == "o" || option == "output")
						checkAndSet([&](const std::string& in) { bb.filename = in; });
					else if (option == "steps")
						checkAndSet([&](const std::string& in) { stage.steps = std::stoi(in); });
					else if (option == "alpha" || option == "a")
						checkAndSet([&](const std::string& in) { stage.alpha = std::stof(in); });
					else if (option == "beta" || option == "b")
						checkAndSet([&](const std::string& in) { stage.beta = std::stof(in); });
					else if (option == "theta" || option == "t")
						checkAndSet([&](const std::string& in) { stage.theta = std::stof(in); });
					else if (option == "escape-trajectories" || option == "et")
						checkAndSet([&](const std::string& in) { bb.escapeThreshold = std::stoi(in); });
					else if (option == "escape-trajectories-red" || option == "etr")
						checkAndSet([&](const std::string& in) { bb.escapeThresholdR = std::stoi(in); });
					else if (option == "escape-trajectories-green" || option == "etg")
						checkAndSet([&](const std::string& in) { bb.escapeThresholdG = std::stoi(in); });
					else if (option == "escape-trajectories-blue" || option == "etb")
						checkAndSet([&](const std::string& in) { bb.escapeThresholdB = std::stoi(in); });
					else if (option == "counter")
						checkAndSet([&](const std::string& in) { bb.counter = std::stoi(in); });
					else if (option == "bezier-enable")
						stage.bezier = true;
					else if (option == "bezier-disable")
						stage.bezier = false;
					else if (option == "random-enable")
						bb.useRandomGeneration = true;
					else if (option == "random-disable")
						bb.useRandomGeneration = false;
					else if (option == "crop-samples-enable")
						bb.cropSamples = true;
					else if (option == "crop-samples-disable")
						bb.cropSamples = false;
					else if (option == "next" || option == "next-stage" || option == "n")
					{
						bb.stages.push_back(stage);
						stage = {};
					}
					else if (option == "next-cpy" || option == "next-stage-copy" || option == "nc")
						bb.stages.push_back(stage);
					else
					{
						LOG("Unknown option: " << arg);
						success = false;
					}
				}
				else
				{
					LOG("Invaid option: " << arg);
					success = false;
				}
			}
			else
			{
				LOG("Invaid argument: " << arg);
				success = false;
			}
		}
		else
		{
			LOG("Empty argument.");
			success = false;
		}
	}

	if (!success)
	{
		LOG("Program exit early.");
		return 1;
	}

	bb.stages.push_back(stage);

	std::srand(time(NULL));

	bb.run();

	LOG("Program exit");

	return 0;
}