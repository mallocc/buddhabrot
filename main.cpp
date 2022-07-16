#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <random>
#include <stack>
#include <functional>

#include "Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Delicious
#define PI 3.1415926f

/**
 *
 * Mersenne Twister random number generator.
 *
 */
float randf(const float range = 1.0f, bool sign = false)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);
	return dis(gen) * range * (1.0f + (float)sign) - range * sign;
}

/**
 * Very basic complex struc with operations.
 */
struct Complex
{
	float re, im;

	Complex() : re(0.0f), im(0.0f) {}
	Complex(float re, float im) : re(re), im(im) {}

	Complex add(const Complex& a) { return Complex(re + a.re, im + a.im); }
	Complex operator+(const Complex& a) { return add(a); }
	Complex sub(const Complex& a) { return Complex(re - a.re, im - a.im); }
	Complex operator-(const Complex& a) { return sub(a); }
	Complex mult(const Complex& a) { return Complex(re * a.re - im * a.im, 
													re * a.im + im * a.re); }
	Complex operator*(const Complex& a) { return mult(a); }

	Complex operator/(float a) { return Complex(re / a, im / a); }
	Complex operator*(float a) { return Complex(re * a, im * a); }

	float mod2() { return re * re + im * im; }

	operator std::string() { return "(" + std::to_string(re) + ", "
										+ std::to_string(im) + "i)"; }
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
		float alpha = 0;
		float beta = 0;
		float theta = 0;
		int steps = 1;
		float gamma = 2;
	};

	BuddhabrotRenderer() {}

	uint8_t* pixelData = nullptr;
	int* buddhaData = nullptr;

	std::string filename = "";

	int width = 0;
	int height = 0;
	int components = 1;
	int samples = 0;
	int iterations = 0;
	int iterationsR = 0;
	int iterationsG = 0;
	int iterationsB = 0;
	float radius = 4.0f;
	bool isAnti = false;
	int escapeThreshold = 0;
	int escapeThresholdR = 0;
	int escapeThresholdG = 0;
	int escapeThresholdB = 0;
	int iterationsMin = 0;

	Complex zr;
	Complex cr;

	int counter = 0;

	std::vector<Stage> stages;

	// This should initialise the data arrays and another 
	// members that's value are derived from other.
	void init()
	{
		components = ( iterationsR > 0 
					|| iterationsG > 0 
					|| iterationsB > 0) ? 3 : 1;

		buddhaData = new int[width * height];
		pixelData = new uint8_t[width * height * components];
	}

	// Zeros all of the buddhaData array
	void clearBuddhaData()
	{
		for (int i = 0; i < width * height; ++i)
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
					  const float alphaL, const float betaL, const float thetaL,
					  const float gamma,
					  const int step)
	{
		bool componentOverride = false;

		if (iterationsR > 0)
		{
			LOG("Processing red channel... ");
			process(buddhaData, 
				width, height, samples, iterationsR, radius, 
				v0, v1, zr, cr,
				alphaL, betaL, thetaL, 
				false, escapeThresholdR, iterationsMin);
			getPixelData(width, height, components, buddhaData, pixelData, 
				gamma, 0);
			clearBuddhaData();
			componentOverride = true;
		}
		if (iterationsG > 0)
		{
			LOG("Processing green channel... ");
			process(buddhaData, width, height, samples, iterationsG, radius, 
				v0, v1, zr, cr,
				alphaL, betaL, thetaL, 
				false, escapeThresholdG, iterationsMin);
			getPixelData(width, height, components, buddhaData, pixelData, 
				gamma, 1);
			clearBuddhaData();
			componentOverride = true;
		}
		if (iterationsB > 0)
		{
			LOG("Processing blue channel... ");
			process(buddhaData, width, height, samples, iterationsB, radius, 
				v0, v1, zr, cr,
				alphaL, betaL, thetaL, 
				false, escapeThresholdB, iterationsMin);
			getPixelData(width, height, components, buddhaData, pixelData, 
				gamma, 2);
			clearBuddhaData();
			componentOverride = true;
		}

		if (!componentOverride)
		{
			process(buddhaData, width, height, samples, iterations, radius, 
				v0, v1, zr, cr,
				alphaL, betaL, thetaL, 
				false, escapeThreshold, iterationsMin);
			getPixelData(width, height, components, buddhaData, pixelData, 
				gamma);
		}

		writeToPNG(filename.empty() ? "" : filename + std::to_string(step),
			width, height, components, pixelData);
	}


	// Linear interpolation
	static float b1(float x0, float x1, float t)
	{
		return x0 + (x1 - x0) * t;
	}

	// Quadratic interpolation
	static float b2(float x0, float x1, float x2, float t)
	{
		return pow(1 - t, 2) * x0 + 2 * (1 - t) * t * x1 + t * t * x2;
	}

	// Cubic interpolation
	static float b3(float x0, float x1, float x2, float x3, float t)
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
		);

		if (stages.size() > 1)
		{
			int stepC = 0;
			for (int stage = 0; stage < stages.size() - 1; ++stage)
				for (int step = 0, steps = stages[stage].steps; step < steps; ++step, ++stepC)
				{
					LOG("Processing stage " << stage << " / " << stages.size() - 2 << ", step " << step << " / " << steps - 1 << "...");
					clearAll();

					float alphaL = stages[stage].alpha;
					float betaL = stages[stage].beta;
					float thetaL = stages[stage].theta;
					Complex v0 = stages[stage].v0;
					Complex v1 = stages[stage].v1;
					float gamma = stages[stage].gamma;

					if (stages.size() > 1)
					{
						float b = b3(0, 0, 1, 1, step / (float)steps); // ease in out
						//float b = b3(0, 1, 1, 1, step / (float)steps); // ease in
						//float b = b3(0, 1, 0, 1, step / (float)steps); // ease out
						alphaL = (b * (stages[stage + 1].alpha - stages[stage].alpha) + stages[stage].alpha) / 180 * PI;
						betaL = (b * (stages[stage + 1].beta - stages[stage].beta) + stages[stage].beta) / 180 * PI;
						thetaL = (b * (stages[stage + 1].theta - stages[stage].theta) + stages[stage].theta) / 180 * PI;
						gamma = (b * (stages[stage + 1].gamma - stages[stage].gamma) + stages[stage].gamma);
						v0.re = (b * (stages[stage + 1].v0.re - stages[stage].v0.re) + stages[stage].v0.re);
						v0.im = (b * (stages[stage + 1].v0.im - stages[stage].v0.im) + stages[stage].v0.im);
						v1.re = (b * (stages[stage + 1].v1.re - stages[stage].v1.re) + stages[stage].v1.re);
						v1.im = (b * (stages[stage + 1].v1.im - stages[stage].v1.im) + stages[stage].v1.im);
					}

					processFrame(v0, v1, zr, cr, alphaL, betaL, thetaL, gamma, stepC + counter);
				}
		}
		else if (!stages.empty())
		{
			clearAll();
			processFrame(stages[0].v0, stages[0].v1, zr, cr, stages[0].alpha / 180 * PI, stages[0].beta / 180 * PI, stages[0].theta / 180 * PI, stages[0].gamma, 0);
		}
	}

	// This is the main buddhabrot algorithm in one function
	static void process(
		int* data, int w, int h, int samples, int iter, int radius = 4.0f, 
		const Complex& minc = Complex(-2, -2), 
		const Complex& maxc = Complex(2, 2),
		const Complex& zr = Complex(), const Complex& cr = Complex(),
		float alpha = 0, float beta = 0, float theta = 0, bool anti = false,
		int threshold = 0, int floorIter = 0, int threadCount = 0)
	{
		// pre commpute //

		// flag for if we are using the minimum threshold
		bool escapeColouring = threshold > 0;

		// find the size of the viewable complex plane
		Complex size = Complex(maxc) - minc;

		// for use in the loop for converting back to screen space
		float cw = w / (size.re);
		float ch = h / (size.im);

		// the center of the viewable complex plane
		Complex center = size / 2.0f + minc;

		// find the OpenMP thread count
		if (threadCount == 0) {
#pragma omp parallel
			{
				threadCount = omp_get_num_threads();
			};
		}

		// leave a core for the rest of the OS
#pragma omp parallel for num_threads(std::max(1, threadCount - 1))
		for (int s = 0; s < samples; ++s)
		{
			// pre allocate potential iteration samples
			Complex* csamples = new Complex[iter];

			// initialise the mandelbrot components
			Complex c(randf() * size.re + minc.re, randf() * size.im + minc.im);
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
						// now apply the rotation matrix on t and c (these are
						// the points on the 4d volume)
						t.re = t.re * cos(alpha + theta) 
								+ c.im * sin(alpha + theta);
						t.im = t.im * cos(beta + theta) 
								- c.re * sin(beta + theta);
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
			//	LOG("Progress: " << std::setprecision(2) << (s / (float)samples) * 100.0f << "%...");
		}
		LOG("Progress: finished!");
	}

	// classic sqrt colouring using gamma correction
	// higher the gamma, the brighter it is
	static uint8_t sqrtColour(float x, float y, float gamma)
	{
		return pow(x / y, 1.0f / gamma) * UCHAR_MAX;
	}

	// normalises the buddhaData into pixelData
	static void getPixelData(int w, int h, int c, int* buddhaData, uint8_t* pixelData, float gamma = 2.0f, int o = -1)
	{
		float maxVal = 1;
		for (int i = 0; i < w * h; ++i)
			maxVal = std::max(maxVal, (float)buddhaData[i]);

		for (int i = 0; i < w * h; ++i)
			for (int cc = 0; cc < c; ++cc)
				if (o == -1 || o == cc)
					pixelData[i * c + cc] = sqrtColour(buddhaData[i], maxVal, gamma);
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

					if (option == "w" || option == "width")
						checkAndSet([&](const std::string& in) { bb.width = std::stoi(in); });
					else if (option == "h" || option == "height")
						checkAndSet([&](const std::string& in) { bb.height = std::stoi(in); });
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
						checkAndSet([&](const std::string& in) { bb.samples = std::stoi(in); });
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