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

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
 //double randf(const double range = 1.0, bool sign = false)
 //{
 //	std::random_device rd;
 //	std::mt19937_64 gen(rd());
 //	std::uniform_real_distribution<double> dis(0.0, 1.0);
 //	return dis(gen) * range * (1.0 + (double)sign) - range * sign;
 //}

double randf(const double minimum = 0.0, const double maximum = 1.0)
{
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_real_distribution<double> dis(0.0, 1.0);
	return dis(gen) * (maximum - minimum) + minimum;
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

std::string secondsToHHMMSS(double duration) {
	std::chrono::seconds totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<double>(duration));
	int hours = totalSeconds.count() / 3600;
	int minutes = (totalSeconds.count() % 3600) / 60;
	int seconds = totalSeconds.count() % 60;

	std::ostringstream oss;
	oss << std::setw(2) << std::setfill('0') << hours << ":";
	oss << std::setw(2) << std::setfill('0') << minutes << ":";
	oss << std::setw(2) << std::setfill('0') << seconds;

	return oss.str();
}

template<class T>
std::string formatDurationToHHMMSS(T& duration) {
	std::chrono::seconds totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
	int hours = totalSeconds.count() / 3600;
	int minutes = (totalSeconds.count() % 3600) / 60;
	int seconds = totalSeconds.count() % 60;

	std::ostringstream oss;
	oss << std::setw(2) << std::setfill('0') << hours << ":";
	oss << std::setw(2) << std::setfill('0') << minutes << ":";
	oss << std::setw(2) << std::setfill('0') << seconds;

	return oss.str();
}

/**
 * Very basic complex struc with operations.
 */
struct Complex
{
	double re, im;

	inline Complex() : re(0.0), im(0.0) {}
	inline Complex(double re, double im) : re(re), im(im) {}

	inline Complex add(const Complex& a) { return Complex(re + a.re, im + a.im); }
	inline Complex operator+(const Complex& a) { return add(a); }
	inline friend Complex operator+(double scalar, const Complex& complex) {
		return Complex(scalar + complex.re, scalar + complex.im);
	}

	inline Complex sub(const Complex& a) { return Complex(re - a.re, im - a.im); }
	inline Complex operator-(const Complex& a) { return sub(a); }
	inline friend Complex operator-(double scalar, const Complex& complex) {
		return Complex(scalar - complex.re, scalar - complex.im);
	}
	inline Complex mult(const Complex& a) {
		return Complex(re * a.re - im * a.im,
			re * a.im + im * a.re);
	}
	inline static Complex mult(const Complex& a, const Complex& b) {
		return Complex(b.re * a.re - b.im * a.im,
			b.re * a.im + b.im * a.re);
	}
	inline Complex mult(const double& a) {
		return Complex(re * a, im * a);
	}
	inline Complex operator*(const Complex& a) { return mult(a); }
	inline Complex operator*(const double& a) { return mult(a); }
	inline Complex operator*(double a) { return Complex(re * a, im * a); }
	inline friend Complex operator*(double scalar, const Complex& complex) {
		return Complex(scalar * complex.re, scalar * complex.im);
	}
	inline friend Complex operator*(const Complex& c1, const Complex& c2) {
		return mult(c1, c2);
	}

	inline Complex operator/(double a) { return Complex(re / a, im / a); }

	// Function to calculate the magnitude (r) of the complex number
	inline double magnitude() const {
		return std::sqrt(re * re + im * im);
	}

	// Function to calculate the argument (theta) of the complex number
	inline double argument() const {
		return std::atan2(im, re);
	}

	// Function to raise a complex number to the power n using De Moivre's theorem
	inline Complex pow(double n) const {
		double r = magnitude();
		double theta = argument();

		double newR = std::pow(r, n);
		double newTheta = n * theta;

		return Complex(newR * std::cos(newTheta), newR * std::sin(newTheta));
	}

	//static Complex abs(const Complex& a) { return { std::abs(a.re), std::abs(a.im) }; }


	inline double mod2() { return re * re + im * im; }

	inline static Complex mandelise(const Complex& z, const Complex& c, const glm::mat2& zm1, const glm::mat2& zm2, const glm::mat2& zm3) {
		glm::vec2 z1(z.re, z.im);
		float xx = z1.x * z1.x;
		float yy = z1.y * z1.y;
		glm::vec2 z2(xx - yy, z1.x * z1.y * 2.0f);
		glm::vec2 z3(xx * z1.x - 3.0f * z1.x * yy, 3.0f * xx * z1.y - yy * z1.y);
		//Complex n;
		//n = n + s1 * (p3 > 0 ? z3.pow(p3) : z3);
		//n = n + s2 * (p2 > 0 ? z2.pow(p2) : z2);
		//n = n + s3 * (p3 > 0 ? z3.pow(p3) : z3);
		//return n + c;
		glm::vec2 n = zm3 * z3 + zm2 * z2 + zm1 * z1 + glm::vec2(c.re, c.im);
		return { n.x, n.y };
		//return z * z + c;
	}

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
		Complex v0 = Complex(-2, -2);
		Complex v1 = Complex(2, 2);
		Complex zt = Complex();
		Complex ct = Complex();
		double alpha = 0;
		double beta = 0;
		double theta = 0;
		double phi = 0;
		int steps = 1;
		double gamma = 2;
		double mhRatio = 0.5;
		bool bezier = false;

		double zScalerC = 0;
		double zAngleC = 0;
		double zYScaleC = 1;

		double zScalerB = 1;
		double zAngleB = 0;
		double zYScaleB = 1;

		double zScalerA = 0;
		double zAngleA = 0;
		double zYScaleA = 1;

		// Linear interpolation
		template <typename T>
		static double lerp(T t, T x0, T x1)
		{
			return x0 + (x1 - x0) * t;
		}

		void lerpTo(double t, const Stage& next)
		{
			alpha = lerp(t, alpha, next.alpha) / 180 * PI;
			beta = lerp(t, beta, next.beta) / 180 * PI;
			theta = lerp(t, theta, next.theta) / 180 * PI;
			phi = lerp(t, phi, next.phi) / 180 * PI;

			gamma = lerp(t, gamma, next.gamma);

			zScalerC = lerp(t, zScalerC, next.zScalerC);
			zAngleC = lerp(t, zAngleC, next.zAngleC);
			zYScaleC = lerp(t, zYScaleC, next.zYScaleC);

			zScalerB = lerp(t, zScalerB, next.zScalerB);
			zAngleB = lerp(t, zAngleB, next.zAngleB);
			zYScaleB = lerp(t, zYScaleB, next.zYScaleB);

			zScalerA = lerp(t, zScalerA, next.zScalerA);
			zAngleA = lerp(t, zAngleA, next.zAngleA);
			zYScaleA = lerp(t, zYScaleA, next.zYScaleA);

			mhRatio = lerp(t, mhRatio, next.mhRatio);
			v0.re = lerp(t, v0.re, next.v0.re);
			v0.im = lerp(t, v0.im, next.v0.im);
			v1.re = lerp(t, v1.re, next.v1.re);
			v1.im = lerp(t, v1.im, next.v1.im);
			zt.re = lerp(t, zt.re, next.zt.re);
			zt.im = lerp(t, zt.im, next.zt.im);
			ct.re = lerp(t, ct.re, next.ct.re);
			ct.im = lerp(t, ct.im, next.ct.im);
		}
	};

	BuddhabrotRenderer() {}

	std::vector<uint8_t> pixelData;
	std::vector<int>  buddhaData;

	std::string filename = "";

	int width = 0;
	int height = 0;
	int components = 1;
	double samples = 0;
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

	int currentStep = 0;
	int currentStage = 0;
	int stepsLeft = 0;
	int totalSteps = 0;

	int currentSamples = 0;

	double bmTime = 0.0f;

	std::vector<Complex> csamples;
	std::vector<double> consamples;

	const int printInterval = 5000000; // Set the interval for printing the current samples

	int sampleTimeMs = 0;

	double benchmark()
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

		buddhaData = std::vector<int>(width * height * superSampleFactor * superSampleFactor, 0);
		pixelData = std::vector<uint8_t>(width * height * components, 0);
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

	// Runs the renderer with the options specified
	void run()
	{
		init();

		bmTime = benchmark();

		LOG(
			"Render details:" << std::endl
			<< "\t " << width << "x" << height << std::endl
			<< "\t samples = " << (useRandomGeneration ? samples : width * samples * height * samples) << std::endl
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
			<< "\t keyframes = " << stages.size() << std::endl
			<< "\t bmTime = " << bmTime << std::endl
		);

		std::cout << "\n\n\n\n\n\n\n";

		for (int stage = 0; stage < stages.size() - 1; ++stage)
			totalSteps += stages[stage].steps;

		if (stages.size() > 1)
		{
			int stepC = 0;
			for (int stage = 0; stage < stages.size() - 1; ++stage)
				for (int step = 0, steps = stages[stage].steps; step < steps; ++step, ++stepC)
				{
					currentStage = stage;
					currentStep = step;
					//print(std::format("Processing stage {} / {}, step {} / {}...", stage, stages.size() - 2, step, steps - 1));
					clearAll();

					Stage tStage = stages[stage];

					if (stage + 1 < stages.size())
					{
						double b = tStage.bezier ? smootherstep(step / (double)steps, 0., 1.) : (step / (double)steps);
						tStage.lerpTo(b, stages[stage + 1]);
					}

					timer.start();

					processFrame(tStage, stepC + counter);

					timer.stop();

					stepsLeft = (totalSteps - stepC);
				}
		}
		else if (!stages.empty())
		{
			clearAll();
			processFrame(stages[0], -1);
		}
	}


	// Processes a single frame with the provided properties
	void processFrame(const Stage& stage, const int step)
	{
		bool componentOverride = false;

		print("Generateing intial samples...");

		generateInitialSamples(std::max<int>(iterationsR, std::max<int>(iterationsG, std::max<int>(iterationsR, iterations))),
			stage);

		if (iterationsR > 0)
		{
			print("Processing red channel... ");
			process(buddhaData,
				width * superSampleFactor, height * superSampleFactor, samples, iterationsR, radius,
				stage, false, escapeThresholdR, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				stage.gamma, 0);
			clearBuddhaData();
			componentOverride = true;
		}
		if (iterationsG > 0)
		{
			print("Processing green channel... ");
			process(buddhaData,
				width * superSampleFactor, height * superSampleFactor, samples, iterationsG, radius,
				stage, false, escapeThresholdR, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				stage.gamma, 1);
			clearBuddhaData();
			componentOverride = true;
		}
		if (iterationsB > 0)
		{
			print("Processing blue channel... ");
			process(buddhaData,
				width * superSampleFactor, height * superSampleFactor, samples, iterationsB, radius,
				stage, false, escapeThresholdR, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				stage.gamma, 2);
			clearBuddhaData();
			componentOverride = true;
		}

		if (!componentOverride)
		{
			process(buddhaData, width * superSampleFactor, height * superSampleFactor, samples, iterations, radius,
				stage, false, escapeThreshold, iterationsMin);
			getPixelData(width, height, components, superSampleFactor, buddhaData, pixelData,
				stage.gamma);
		}

		writeToPNG(filename.empty() ? "" : filename + (step == -1 ? "" : std::to_string(step)),
			width, height, components, pixelData);
	}

	void clearLastLines(int n) {
		for (int i = 0; i < n; i++) {
			std::cout << "\033[2K";  // Clear the current line
			std::cout << "\033[A";   // Move the cursor up
		}
	}

	int countLinesInStringStream(std::stringstream& ss) {
		std::string line;
		int lineCount = 0;

		// Save the current position in the stream
		std::streampos originalPos = ss.tellg();

		// Count the lines by reading the stream line by line
		while (std::getline(ss, line)) {
			lineCount++;
		}

		// Restore the original position in the stream
		ss.clear();
		ss.seekg(originalPos);

		return lineCount;
	}


	std::string drawProgressBar(double progress) {
		int barWidth = 30;
		int pos = static_cast<int>(barWidth * progress);

		std::stringstream ss;
		ss << "[";
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos)
				ss << "=";
			else if (i == pos)
				ss << ">";
			else
				ss << " ";
		}
		ss << "] " << static_cast<int>(std::round(progress * 100.0f)) << "%";
		return ss.str();
	}

	std::string lastMessage;

	void print(const std::string& str)
	{
		std::string totalProgress = drawProgressBar(1 - stepsLeft / (double)totalSteps);
		std::string frameProgress = drawProgressBar(currentSamples / (double)(useRandomGeneration ? samples : width * height * samples * samples));
		std::stringstream ss;
		ss << "\n          Average time: " << timer.getAverageTime() / 1000.0f << "s"
			<< "\n        Est. time left: " << secondsToHHMMSS(stepsLeft * timer.getAverageTime() / 1000.0f)
			<< "\n  Est. frame time left: " << secondsToHHMMSS(((samples - currentSamples) / (double)printInterval) * sampleTimeMs / 1000.0f)
			<< (stages.size() > 1 ? std::format("\n            Processing: STAGE {} / {},\tSTEP {} / {}", currentStage + 1, stages.size() - 1, currentStep + 1, stages[currentStage].steps) : "")
			<< "\n        Current Sample: " << currentSamples
			<< "\n             Currently: " << (str.empty() ? lastMessage : str)
			<< "\n        Frame Progress:" << frameProgress
			<< "\n        Total Progress:" << (stages.size() == 1 ? frameProgress : totalProgress);
		clearLastLines(countLinesInStringStream(ss));
		std::cout << ss.str() << std::endl;
		if (!str.empty())
			lastMessage = str;
	}


	// map dimension name to index
	std::map<std::string, int> dimensions = {
		{"zr", 0},
		{"zi", 1},
		{"cr", 2},
		{"ci", 3}
	};

	Complex mutate(Complex& c, Complex& size, const Complex& minc, const Complex& maxc, double threshold = 0.5)
	{
		if (randf(0, 1) < threshold)
		{
			Complex n = c;

			double zoom = 4.0f / size.re;
			double phi = randf(0, 6.28318530718);
			double r = 0.01 / zoom;
			r *= randf(0, 1);

			n.re += r * cos(phi);
			n.im += r * sin(phi);

			return n;
		}
		else
		{
			c = { randf(minc.re, maxc.re), randf(minc.im, maxc.im) };
			//c = { randf(-2., 2.), randf(-2., 2.) };
			return c;
		}

	}

	double contrib(int len, std::vector<Complex>& orbit, const Complex& minc, const Complex& maxc)
	{
		double contrib = 0;
		int inside = 0, i;

		for (i = 0; i < len; i++)
			if (orbit[i].re >= minc.re && orbit[i].re < maxc.re && orbit[i].im >= minc.im && orbit[i].im < maxc.im)
				contrib++;

		return contrib / double(len);
	}

	double TransitionProbability(double q1, double q2, double olen1, double olen2)
	{
		return (1.f - (q1 - olen1) / q1) / (1.f - (q2 - olen2) / q2);
	}

	glm::mat2 buildMatrix(float theta, float scaler, float yscale)
	{
		float thetaRadians = theta * glm::pi<float>() / 180.0f;
		float cosTheta = std::cos(thetaRadians);
		float sinTheta = std::sin(thetaRadians);

		return glm::mat2(
			cosTheta * scaler, sinTheta * scaler * yscale,
			-sinTheta * scaler, cosTheta * scaler * yscale
		);
	}

	void generateInitialSamples(int iter, const Stage& stage)
	{
		// find the size of the viewable complex plane
		Complex size = Complex(stage.v1) - stage.v0;

		// the center of the viewable complex plane
		Complex center = size / 2.0 + stage.v0;

		glm::mat2 zm1 = buildMatrix(stage.zAngleA, stage.zScalerA, stage.zYScaleA);
		glm::mat2 zm2 = buildMatrix(stage.zAngleB, stage.zScalerB, stage.zYScaleB);
		glm::mat2 zm3 = buildMatrix(stage.zAngleC, stage.zScalerC, stage.zYScaleC);

		auto evalOrbit = [&](std::vector<Complex>& orbit, int& i, Complex& c) {
			Complex z(c);

			// we only care about trajetories that are less than max iterations 
			// in length and that they fall within the radius bounds
			for (i = 0; i < iter && z.mod2() < radius; ++i)
			{
				// translations through the shape (there are 4 axes)
				z = z + zr;
				c = c + cr;
				// apply the magic formula
				//z = z * z + c;

				z = z.mandelise(z, c, zm1, zm2, zm3);
				// store our sample complex position for later
				orbit[i] = z;
			}

			if (z.mod2() > radius)
				return true;

			return false;
		};

		std::function<bool(std::vector<Complex>&, Complex&, double, double, double, int)> FindInitialSample =
			[&](std::vector<Complex>& orbit, Complex& c, double x, double y, double rad, int f) -> bool
		{
			if (f > 150)
				return false;

			Complex ct = c, tmp, seed;

			int m = -1, i;
			double closest = 1e20;

			for (i = 0; i < 150; i++)
			{
				tmp = { randf(-rad, rad), randf(-rad, rad) };
				tmp.re += x;
				tmp.im += y;
				int orbitLen = 0;
				if (!evalOrbit(orbit, orbitLen, tmp))
					continue;

				if (contrib(orbitLen, orbit, stage.v0, stage.v1) > 0.0f)
				{
					c = tmp;
					return true;
				}

				for (int q = 0; q < orbit.size(); q++)
				{
					double d = (orbit[q] - center).mod2();

					if (d < closest)
						m = q,
						closest = d,
						seed = tmp;
				}
			}

			return FindInitialSample(orbit, c, seed.re, seed.im, rad/2.0f, f + 1);
		};

		if (csamples.empty())
		{
			csamples = {};
			consamples = {};

			
			for (int maxTries = 0; csamples.empty() && maxTries < 100; ++maxTries)
			{
#ifndef _DEBUG
#pragma omp parallel num_threads(std::max(1, omp_get_num_threads() - 1))
#endif
				{
					std::vector<Complex> tempOrbit(iter);
#ifndef _DEBUG
#pragma omp for
#endif
					for (int e = 0; e < 30; ++e)
					{
						Complex m;
						if (FindInitialSample(tempOrbit, m, 0, 0, radius / 2.0f, 0))
						{
							int orbitLen = 0;
							evalOrbit(tempOrbit, orbitLen, m);
#ifndef _DEBUG
#pragma omp critical
#endif
							{
								csamples.push_back(m);
								consamples.push_back(contrib(orbitLen, tempOrbit, stage.v0, stage.v1));
							}
						}
					}
				}
			}
		}
		else
		{
#ifndef _DEBUG
#pragma omp parallel num_threads(std::max(1, omp_get_num_threads() - 1))
#endif
			{
				std::vector<Complex> tempOrbit(iter);
#pragma omp for
				for (int e = 0; e < csamples.size(); ++e)
				{
					Complex m = mutate(csamples[e], size, stage.v0, stage.v1, stage.mhRatio);
					int orbitLen = 0;
					evalOrbit(tempOrbit, orbitLen, m);
					int con = contrib(orbitLen, tempOrbit, stage.v0, stage.v1);
					if (consamples[e] < con)
					{
						csamples[e] = m;
						consamples[e] = con;
					}
				}
			}
		}
	}


	// Function to create a 4D rotation matrix given angles alpha, beta, theta, and phi
	glm::mat4 create4DRotationMatrix(double alpha, double beta, double theta, double phi) {
		glm::quat quaternion = glm::quat(glm::vec3(theta, beta, alpha));

		// Convert the quaternion to a 4x4 rotation matrix
		glm::mat4 rotationMatrix = glm::mat4_cast(quaternion);

		return rotationMatrix;
	}

	glm::mat4 rotation_zxcx(double alpha, double beta, double theta, double phi) {
		// Rotation around the z-axis
		glm::mat4 rotation_z = create4DRotationMatrix(0.0, 0.0, theta, 0.0);

		// Rotation around the x-axis
		glm::mat4 rotation_x1 = create4DRotationMatrix(alpha, 0.0, 0.0, 0.0);

		// Rotation around the x-axis (second time)
		glm::mat4 rotation_x2 = create4DRotationMatrix(-beta, 0.0, 0.0, 0.0);

		// Combine the rotations in the specified order (z * x1 * x2)
		glm::mat4 rotationMatrix = rotation_z * rotation_x1 * rotation_x2;

		return rotationMatrix;
	}

	glm::mat4 rotation_zxcy(double alpha, double beta, double theta, double phi) {
		// Rotation around the z-axis
		glm::mat4 rotation_z = create4DRotationMatrix(0.0, 0.0, theta, 0.0);

		// Rotation around the x-axis
		glm::mat4 rotation_x = create4DRotationMatrix(alpha, 0.0, 0.0, 0.0);

		// Rotation around the y-axis
		glm::mat4 rotation_y = create4DRotationMatrix(0.0, beta, 0.0, 0.0);

		// Combine the rotations in the specified order (z * x * y)
		glm::mat4 rotationMatrix = rotation_z * rotation_x * rotation_y;

		return rotationMatrix;
	}

	glm::mat4 rotation_zycx(double alpha, double beta, double theta, double phi) {
		// Rotation around the z-axis
		glm::mat4 rotation_z = create4DRotationMatrix(0.0, 0.0, theta, 0.0);

		// Rotation around the y-axis
		glm::mat4 rotation_y = create4DRotationMatrix(0.0, beta, 0.0, 0.0);

		// Rotation around the x-axis
		glm::mat4 rotation_x = create4DRotationMatrix(alpha, 0.0, 0.0, 0.0);

		// Combine the rotations in the specified order (z * y * x)
		glm::mat4 rotationMatrix = rotation_z * rotation_y * rotation_x;

		return rotationMatrix;
	}

	glm::mat4 rotation_zycy(double alpha, double beta, double theta, double phi) {
		// Rotation around the z-axis
		glm::mat4 rotation_z = create4DRotationMatrix(0.0, 0.0, theta, 0.0);

		// Rotation around the y-axis (first time)
		glm::mat4 rotation_y1 = create4DRotationMatrix(0.0, beta, 0.0, 0.0);

		// Rotation around the y-axis (second time)
		glm::mat4 rotation_y2 = create4DRotationMatrix(0.0, -phi, 0.0, 0.0);

		// Combine the rotations in the specified order (z * y1 * y2)
		glm::mat4 rotationMatrix = rotation_z * rotation_y1 * rotation_y2;

		return rotationMatrix;
	}

	// This is the main buddhabrot algorithm in one function
	void process(
		std::vector<int>& data, int w, int h, double samples, int iter, int radius,
		const Stage& stage,
		bool anti = false,
		int threshold = 0, int floorIter = 0, int threadCount = 0)
	{
		// pre commpute //

		auto rotationMatrix = create4DRotationMatrix(stage.alpha, stage.beta, stage.theta, stage.phi);
		auto zxcxRotationMatrix = rotation_zxcx(stage.alpha, stage.beta, stage.theta, stage.phi);
		auto zxcyRotationMatrix = rotation_zxcy(stage.alpha, stage.beta, stage.theta, stage.phi);
		auto zycxRotationMatrix = rotation_zycx(stage.alpha, stage.beta, stage.theta, stage.phi);
		auto zycyRotationMatrix = rotation_zycy(stage.alpha, stage.beta, stage.theta, stage.phi);

		// flag for if we are using the minimum threshold
		bool escapeColouring = threshold > 0;

		// find the size of the viewable complex plane
		Complex size = Complex(stage.v1) - stage.v0;

		// the center of the viewable complex plane
		Complex center = size / 2.0 + stage.v0;

		// for use in the loop for converting back to screen space
		double cw = w / (size.re);
		double ch = h / (size.im);

		std::vector<double> coords = { 0.0f, 0.0f, 0.0f, 0.0f };

		double incw = size.re / (double)(w * samples); // Width of each sample
		double inch = size.im / (double)(h * samples); // Height of each sample

		double incwF = 4.0f / (double)(w * samples); // Width of each sample 
		double inchF = 4.0f / (double)(h * samples); // Height of each sample

		int wsamples = w * samples;
		int hsamples = h * samples;

		glm::mat2 zm1 = buildMatrix(stage.zAngleA, stage.zScalerA, stage.zYScaleA);
		glm::mat2 zm2 = buildMatrix(stage.zAngleB, stage.zScalerB, stage.zYScaleB);
		glm::mat2 zm3 = buildMatrix(stage.zAngleC, stage.zScalerC, stage.zYScaleC);

		auto trajectory = [&](std::vector<int>& localData, std::vector<Complex>& orbit, Complex& bestC, double& bestCon) {
			// initialise the mandelbrot components
			Complex c = mutate(bestC, size, stage.v0, stage.v1, stage.mhRatio);

			Complex z(c);

			bool goodOrbit = false;

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
				//z = z * z + c;

				z = z.mandelise(z, c, zm1, zm2, zm3);

				// store our sample complex position for later
				orbit[i] = z;

				// track orbits that have entered the screen
				if (!goodOrbit)
					if (orbit[i].re >= stage.v0.re && orbit[i].re < stage.v1.re && orbit[i].im >= stage.v0.im && orbit[i].im < stage.v1.im)
						goodOrbit = true;
			}

			// filter for minimum iterations
			// flags to check between normal and anti brot
			if (goodOrbit)
				if (i >= floorIter && ((!anti && i < iter) || (anti && i == iter)))
				{
					double contribution = contrib(i, orbit, stage.v0, stage.v1);

					//double T1 = TransitionProbability(iter, l, i, o);
					//double T2 = TransitionProbability(l, iter, o, i);

					//double alpha = std::min<double>(1.f, std::exp(std::log(contribution * T1) - std::log(bestCon * T2)));
					//double R = randf();

					bestCon = contribution;
					bestC = c;

					// if we want to rotate around a point, we must translate the point
					// to the origin first (we will do it for Z later, remember 4 axes)
					c = c - center;

					// iterate through our valid iterations samples
					for (int j = 0; j < i; ++j)
					{
						// rotate around current center point //
						Complex& t = orbit[j];

						// if we want to rotate around a point, 
						// we must translate the point to the origin first
						t = t - center;

						coords[0] = t.re;
						coords[1] = t.im;
						coords[2] = c.re;
						coords[3] = c.im;

						double x1 = coords[volumeAX];
						double y1 = coords[volumeAY];
						double z1 = coords[volumeAZ];
						double x2 = coords[volumeBX];
						double y2 = coords[volumeBY];
						double z2 = coords[volumeBZ];

						double nx = x1 + (x2 - x1) * stage.alpha;
						double ny = y1 + (y2 - y1) * stage.beta;
						double nz = z1 + (z2 - z1) * stage.theta;

						// Convert 3D points to 4D points (adding w component)
						double nw = 1.0;
						glm::vec4 point4D(nx, ny, nz, nw);

						// Apply the rotation matrix on the 4D point
						glm::vec4 rotatedPoint = rotationMatrix * point4D;
						//glm::vec4 rotatedPoint = zycyRotationMatrix * zycxRotationMatrix * zxcyRotationMatrix * zxcxRotationMatrix * point4D;

						// Convert the 4D point back to 3D
						double newX = rotatedPoint.x / rotatedPoint.w;
						double newY = rotatedPoint.y / rotatedPoint.w;
						double newZ = rotatedPoint.z / rotatedPoint.w;

						// Update the point coordinates
						t.re = newX;
						t.im = newY;

						// translate back to our point of interest
						t = t + center;

						// transform complex point into screen space point 
						// (screen space coord)
						int x = (t.re - stage.v0.re) * cw;
						int y = (t.im - stage.v0.im) * ch;
						// make sure it falls within in the screen buffer
						if (x >= 0 && x < w && y >= 0 && y < h)
							// incr each pixels components according to the 
							// colour thresholds
							localData[(y * w + x)] += escapeColouring
							? j >= threshold
							: j < iter;
					}
				}

		};
		auto trajectory1 = [&](std::vector<int>& localData, std::vector<Complex>& orbit, int s, int px = 0, int py = 0) {
			// initialise the mandelbrot components
			Complex c;
			if (cropSamples)
				c = Complex(px * incw, py * inch);
			else
				c = Complex(px * incwF, py * inchF);
			c = c + stage.v0;

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
				//z = z * z + c;
				z = z.mandelise(z, c, zm1, zm2, zm3);

				// store our sample complex position for later
				orbit[i] = z;
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
						Complex& t = orbit[j];

						// if we want to rotate around a point, 
						// we must translate the point to the origin first
						t = t - center;

						coords[0] = t.re;
						coords[1] = t.im;
						coords[2] = c.re;
						coords[3] = c.im;

						double x1 = coords[volumeAX];
						double y1 = coords[volumeAY];
						double z1 = coords[volumeAZ];
						double x2 = coords[volumeBX];
						double y2 = coords[volumeBY];
						double z2 = coords[volumeBZ];

						double nx = x1 + (x2 - x1) * stage.alpha;
						double ny = y1 + (y2 - y1) * stage.beta;
						double nz = z1 + (z2 - z1) * stage.theta;

						// Convert 3D points to 4D points (adding w component)
						double nw = 1.0;
						glm::vec4 point4D(nx, ny, nz, nw);

						// Apply the rotation matrix on the 4D point
						glm::vec4 rotatedPoint = rotationMatrix * point4D;
						//glm::vec4 rotatedPoint = zycyRotationMatrix * zycxRotationMatrix * zxcyRotationMatrix * zxcxRotationMatrix * point4D;

						// Convert the 4D point back to 3D
						double newX = rotatedPoint.x / rotatedPoint.w;
						double newY = rotatedPoint.y / rotatedPoint.w;
						double newZ = rotatedPoint.z / rotatedPoint.w;

						// Update the point coordinates
						t.re = newX;
						t.im = newY;

						// translate back to our point of interest
						t = t + center;

						// transform complex point into screen space point 
						// (screen space coord)
						int x = (t.re - stage.v0.re) * cw;
						int y = (t.im - stage.v0.im) * ch;
						// make sure it falls within in the screen buffer
						if (x >= 0 && x < w && y >= 0 && y < h)
							// incr each pixels components according to the 
							// colour thresholds
							localData[(y * w + x)] += escapeColouring
							? j >= threshold
							: j < iter;
					}

		};

		// find the OpenMP thread count
		if (threadCount == 0 || threadCount > omp_get_num_threads())
#pragma omp parallel
			threadCount = omp_get_num_threads();

		// Allocate thread-local arrays to store intermediate results for each thread
		std::vector<std::vector<int>> threadLocalData(threadCount, std::vector<int>(w * h, 0));

		Timer<std::chrono::milliseconds> localTimer;
		localTimer.start();

		currentSamples = 0;
#ifndef _DEBUG
#pragma omp parallel num_threads(std::max(1, threadCount - 1))
#endif
		{
			int threadId = omp_get_thread_num();
			// pre allocate potential iteration samples
			std::vector<Complex> orbit(iter);

			std::vector<Complex> localCSamples = csamples;
			std::vector<double> localCon = consamples;

			int thisSamples = 0;


			if (useRandomGeneration)
#ifndef _DEBUG
#pragma omp for
#endif
				for (int s = 0; s < (int)samples; ++s)
				{
					trajectory(threadLocalData[threadId], orbit, localCSamples[s % localCSamples.size()], localCon[s % localCon.size()]);
					//trajectory(threadLocalData[threadId], orbit, s);
#ifndef _DEBUG
#pragma omp atomic
#endif
					currentSamples += 1;
					if (s % printInterval == 0)
					{
#ifndef _DEBUG
#pragma omp critical
#endif
						{
							localTimer.stop();
							sampleTimeMs = localTimer.getAverageTime();
							print("");
							localTimer.start();
						}
					}
				}
			else
#pragma omp for
				for (int px = 0; px < wsamples; ++px)
					for (int py = 0; py < hsamples; ++py)
					{
						trajectory1(threadLocalData[threadId], orbit, 0, px, py);
						//#ifndef _DEBUG
						//#pragma omp atomic
						//#endif
						//						currentSamples += 1;
						//						if ((px * py) % printInterval == 0) {
						//#ifndef _DEBUG
						//#pragma omp critical
						//#endif
						//							print("");
						//						}
					}

#ifndef _DEBUG
#pragma omp critical
#endif
			{
				csamples = localCSamples;
				consamples = localCon;
			}
		}

		// Combine the thread-local data into the final 'data' array
		for (int threadId = 0; threadId < threadLocalData.size(); ++threadId)
			for (int i = 0; i < w * h; ++i)
				data[i] += threadLocalData[threadId][i];

		print("Progress: finished!");
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

	static double smootherstep(double x, double minVal, double maxVal)
	{
		return smoothstep(smoothstep(x, minVal, maxVal), minVal, maxVal);
	}


	// Apply box blur to a 2D array
	static void boxBlur2D(int w, int h, int superSampleFactor, std::vector<int>& input, double* output, int radius)
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
				output[y * w * superSampleFactor + x] = sum / (double)kernelSize;
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
				output[y * w * superSampleFactor + x] = sum / (double)kernelSize;
			}
		}
	}

	// Apply medium filter to a 2D array
	static void mediumFilter2D(int w, int h, std::vector<int>& input, int radius, int threshold = 0)
	{
		int kernelSize = 2 * radius + 1;
		std::vector<int> kernel(kernelSize, 1);

		// Set the weight for the center pixel in the kernel to 2
		kernel[radius] = 2;

		std::vector<int> output(w * h, 0);

		std::vector<int> values(kernelSize);

		// Blur horizontally
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				for (int i = -radius; i <= radius; ++i)
				{
					int px = x + i;
					if (px < 0) px = 0;
					if (px >= w) px = w - 1;

					values[i + radius] = (input[y * w + px]);
				}
				std::sort(values.begin(), values.end());

				int median = values[kernelSize / 2];

				output[y * w + x] += abs(input[y * w + x] - median) > threshold ? median : input[y * w + x];
			}
		}

		// Blur vertically
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				for (int i = -radius; i <= radius; ++i)
				{
					int py = y + i;
					if (py < 0) py = 0;
					if (py >= h) py = h - 1;

					values[i + radius] = (input[py * w + x]);
				}
				std::sort(values.begin(), values.end());

				int median = values[kernelSize / 2];

				output[y * w + x] += abs(input[y * w + x] - median) > threshold ? median : input[y * w + x];
			}
		}

		for (auto& i : output)
			i /= 2.0f;

		input = output;
	}

	// Apply medium filter to a 2D array
	static void filter2D(int w, int h, std::vector<int>& input, int radius)
	{
		int kernelSize = 2 * radius + 1;
		std::vector<int> kernel(kernelSize, 1);

		// Set the weight for the center pixel in the kernel to 2
		kernel[radius] = 2;

		std::vector<int> output(w * h, 0);

		std::vector<int> values(kernelSize);

		// Blur horizontally
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				int left = 0, right = 0, center = input[y * w + x];
				if (x > 0)
					left = input[y * w + x - 1];
				if (x > 0)
					right = input[y * w + x + 1];

			}
		}

		// Blur vertically
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				for (int i = -radius; i <= radius; ++i)
				{
					int py = y + i;
					if (py < 0) py = 0;
					if (py >= h) py = h - 1;

					values[i + radius] = (input[py * w + x]);
				}
				std::sort(values.begin(), values.end());
				output[y * w + x] += values[kernelSize / 2];
			}
		}

		for (auto& i : output)
			i /= 2.0f;

		input = output;
	}

	// Normalizes the buddhaData into pixelData with supersampling and box blur
	static void getPixelData(int w, int h, int c, int superSampleFactor, std::vector<int>& buddhaData, std::vector<uint8_t>& pixelData, double gamma = 2.0, int o = -1)
	{
		int supersampledW = w * superSampleFactor;
		int supersampledH = h * superSampleFactor;

		// Create a temporary buffer for supersampled data
		std::vector<double> supersampledBuddhaData(supersampledW * supersampledH);

		// Apply box blur to the supersampled buddhaData
		if (superSampleFactor > 1)
			boxBlur2D(w, h, superSampleFactor, buddhaData, supersampledBuddhaData.data(), 1);


		//mediumFilter2D(w, h, buddhaData, 1, 0);

		// Calculate maximum value in the supersampled image
		// Find the minimum and maximum values
		double minValue = *std::min_element(buddhaData.data(), buddhaData.data() + buddhaData.size());
		double maxValue = *std::max_element(buddhaData.data(), buddhaData.data() + buddhaData.size());
		// Subtract the minimum value and divide by the range
		double range = maxValue - minValue;

		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				double newValue = buddhaData[y * w + x];
				// Downscale the supersampled buddhaData to pixelData
				if (superSampleFactor > 1)
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
					averagedValue /= (double)(superSampleFactor * superSampleFactor);
				}

				for (int cc = 0; cc < c; ++cc)
				{
					if (o == -1 || o == cc)
					{
						pixelData[(y * w + x) * c + cc] = pow(smoothstep((newValue - minValue) / range, 0.0f, 1.0f), 1.0 / gamma) * UCHAR_MAX;
					}
				}
			}
		}
	}

	// normalises the buddhaData into pixelData
	static void getPixelData2(int w, int h, int c, int superSampleFactor, std::vector<int>& buddhaData, std::vector<uint8_t>& pixelData, double gamma = 2.0, int o = -1)
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
	void writeToPNG(const std::string& filename, int w, int h, int c, std::vector<uint8_t>& data)
	{
		print("Writing out render to PNG image...");
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
		stbi_write_png(ss.str().c_str(), w, h, c, data.data(), w * c);
	}

};

int main(int argc, char* argv[])
{
	LOG("Program entry");

	std::stack<std::string> args;
	for (int a = argc - 1; a >= 1; --a)
		args.push(argv[a]);

	for (int a = 0; a < argc; ++a)
		std::cout << argv[a] << " ";
	std::cout << std::endl;

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
							if (!success)
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

					else if (option == "ztr")
						checkAndSet([&](const std::string& in) { stage.zt.re = std::stof(in); });
					else if (option == "zti")

						checkAndSet([&](const std::string& in) { stage.zt.im = std::stof(in); });
					else if (option == "ctr")
						checkAndSet([&](const std::string& in) { stage.ct.re = std::stof(in); });
					else if (option == "cti")

						checkAndSet([&](const std::string& in) { stage.ct.im = std::stof(in); });
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
					else if (option == "phi" || option == "p")
						checkAndSet([&](const std::string& in) { stage.phi = std::stof(in); });

					else if (option == "zScalerA" || option == "zsa")
						checkAndSet([&](const std::string& in) { stage.zScalerA = std::stof(in); });
					else if (option == "zScalerB" || option == "zsb")
						checkAndSet([&](const std::string& in) { stage.zScalerB = std::stof(in); });
					else if (option == "zScalerC" || option == "zsc")
						checkAndSet([&](const std::string& in) { stage.zScalerC = std::stof(in); });

					else if (option == "zAngleA" || option == "zaa")
						checkAndSet([&](const std::string& in) { stage.zAngleA = std::stof(in); });
					else if (option == "zAngleB" || option == "zab")
						checkAndSet([&](const std::string& in) { stage.zAngleB = std::stof(in); });
					else if (option == "zAngleC" || option == "zac")
						checkAndSet([&](const std::string& in) { stage.zAngleC = std::stof(in); });

					else if (option == "zYScaleA" || option == "zysa")
						checkAndSet([&](const std::string& in) { stage.zYScaleA = std::stof(in); });
					else if (option == "zYScaleB" || option == "zysb")
						checkAndSet([&](const std::string& in) { stage.zYScaleB = std::stof(in); });
					else if (option == "zYScaleC" || option == "zysc")
						checkAndSet([&](const std::string& in) { stage.zYScaleC = std::stof(in); });

					else if (option == "mhRatio")
						checkAndSet([&](const std::string& in) { stage.mhRatio = std::stof(in); });

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

	Timer<std::chrono::milliseconds> totalTime;
	totalTime.start();

	bb.run();

	totalTime.stop();

	LOG("Total time: " << secondsToHHMMSS(convertToSeconds<std::chrono::milliseconds>(totalTime.getAverageTime())));

	LOG("Program exit");

	return 0;
}