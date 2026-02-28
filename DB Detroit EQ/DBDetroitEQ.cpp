#include "mp_sdk_audio.h"
#include <cmath>
#include <algorithm>
#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif
//test5
using namespace gmpi;

class DBDetroitEQ final : public MpBase2
{
	AudioInPin pinInput;
	AudioInPin pinFreq;
	AudioInPin pinGain;
	AudioOutPin pinOutput;

	// Biquad filter state variables
	float z1 = 0.0f;
	float z2 = 0.0f;

	// Biquad coefficients (normalized)
	float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
	float a1 = 0.0f, a2 = 0.0f;

	// Parameter smoothing
	float targetFreq = 0.0f;
	float targetGain = 0.0f;

	float smoothedFreq = 0.0f;
	float smoothedGain = 0.0f;

	float currentFreq = -1.0f;
	float currentGain = -1.0f;

	float cachedSampleRate = 0.0f;
	float cachedNyquistFreq = 0.0f;
	float smoothingCoeff = 0.0f;

	// Constants
	static constexpr float PI = 3.14159265359f;
	static constexpr float EPSILON = 1e-6f;
	static constexpr float BASE_Q = 0.08f;
	static constexpr float GAIN_THRESHOLD = 1.3f;
	static constexpr float Q_SLOW_RATE = 20.0f;
	static constexpr float Q_NORMAL_RATE = 8.0f;
	static constexpr float MIN_FREQ = 10.0f;
	static constexpr float MAX_FREQ = 20000.0f;
	static constexpr float SMOOTHING_TIME_MS = 0.999f; // 0.999 ms smoothing time

	inline bool hasChanged(float newVal, float oldVal) const
	{
		return std::abs(newVal - oldVal) > EPSILON;
	}

	void enableDenormalHandling()
	{
#if defined(_MSC_VER) || defined(__GNUC__)
		// Enable Flush-To-Zero and Denormals-Are-Zero modes for SSE
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
	}

	void updateSampleRateCache()
	{
		const float sampleRate = getSampleRate();
		if (cachedSampleRate != sampleRate)
		{
			cachedSampleRate = sampleRate;
			cachedNyquistFreq = sampleRate * 0.5f;
			// Calculate smoothing coefficient: coeff = 1 - exp(-1 / (time * sampleRate))
			smoothingCoeff = 1.0f - std::exp(-1.0f / (SMOOTHING_TIME_MS * 0.001f * sampleRate));
		}
	}

	void calculateCoefficients(float frequency, float gainDb, float& b0, float& b1, float& b2, float& a1, float& a2)
	{
		// Multiply frequency by 10
		frequency *= 10.0f;

		// Multiply gain by 10
		gainDb *= 10.0f;

		// Clamp frequency to valid range
		const float maxFreq = (std::min)(MAX_FREQ, cachedNyquistFreq);
		frequency = std::clamp(frequency, MIN_FREQ, maxFreq);

		const float A = std::pow(10.0f, gainDb / 40.0f);
		const float gainMagnitude = std::abs(gainDb);

		// Proportional Q - Motown-style curves
		const float Q = (gainMagnitude <= GAIN_THRESHOLD)
			? BASE_Q + (gainMagnitude / Q_SLOW_RATE)
			: BASE_Q + (GAIN_THRESHOLD / Q_SLOW_RATE) + ((gainMagnitude - GAIN_THRESHOLD) / Q_NORMAL_RATE);

		const float omega = 2.0f * PI * frequency / cachedSampleRate;
		const float sinOmega = std::sin(omega);
		const float cosOmega = std::cos(omega);
		const float alpha = sinOmega / (2.0f * Q);

		// Bell filter coefficients
		const float alphaA = alpha * A;
		const float alphaInvA = alpha / A;
		const float negTwoCosOmega = -2.0f * cosOmega;

		const float b0_temp = 1.0f + alphaA;
		const float b2_temp = 1.0f - alphaA;
		const float a0_temp = 1.0f + alphaInvA;
		const float a2_temp = 1.0f - alphaInvA;

		// Normalize coefficients
		const float invA0 = 1.0f / a0_temp;
		b0 = b0_temp * invA0;
		b1 = negTwoCosOmega * invA0;
		b2 = b2_temp * invA0;
		a1 = negTwoCosOmega * invA0;
		a2 = a2_temp * invA0;
	}

public:
	DBDetroitEQ()
	{
		initializePin(pinInput);
		initializePin(pinFreq);
		initializePin(pinGain);
		initializePin(pinOutput);
		enableDenormalHandling();
	}

	void subProcess(int sampleFrames)
	{
		auto input = getBuffer(pinInput);
		auto freq = getBuffer(pinFreq);
		auto gain = getBuffer(pinGain);
		auto output = getBuffer(pinOutput);

		// Update target values from pin inputs
		targetFreq = freq[0];
		targetGain = gain[0];

		// Smooth parameters towards target
		smoothedFreq += (targetFreq - smoothedFreq) * smoothingCoeff;
		smoothedGain += (targetGain - smoothedGain) * smoothingCoeff;

		// Check if smoothed parameters changed enough to recalculate coefficients
		const bool paramsChanged = hasChanged(smoothedFreq, currentFreq) || hasChanged(smoothedGain, currentGain);

		if (paramsChanged)
		{
			currentFreq = smoothedFreq;
			currentGain = smoothedGain;
			calculateCoefficients(currentFreq, currentGain, b0, b1, b2, a1, a2);
		}

		// Cache coefficients in local variables for better register allocation
		const float c_b0 = b0;
		const float c_b1 = b1;
		const float c_b2 = b2;
		const float c_a1 = a1;
		const float c_a2 = a2;

		float local_z1 = z1;
		float local_z2 = z2;

		// Process with loop unrolling for better performance
		int s = 0;
		const int unrollCount = sampleFrames & ~3;

		for (; s < unrollCount; s += 4)
		{
			// Processes 4 samples per iteration

			// Unrolled iteration 0
			const float inputSample0 = input[s];
			const float outputSample0 = c_b0 * inputSample0 + local_z1;
			local_z1 = c_b1 * inputSample0 - c_a1 * outputSample0 + local_z2;
			local_z2 = c_b2 * inputSample0 - c_a2 * outputSample0;
			output[s] = outputSample0;

			// Unrolled iteration 1
			const float inputSample1 = input[s + 1];
			const float outputSample1 = c_b0 * inputSample1 + local_z1;
			local_z1 = c_b1 * inputSample1 - c_a1 * outputSample1 + local_z2;
			local_z2 = c_b2 * inputSample1 - c_a2 * outputSample1;
			output[s + 1] = outputSample1;

			// Unrolled iteration 2
			const float inputSample2 = input[s + 2];
			const float outputSample2 = c_b0 * inputSample2 + local_z1;
			local_z1 = c_b1 * inputSample2 - c_a1 * outputSample2 + local_z2;
			local_z2 = c_b2 * inputSample2 - c_a2 * outputSample2;
			output[s + 2] = outputSample2;

			// Unrolled iteration 3
			const float inputSample3 = input[s + 3];
			const float outputSample3 = c_b0 * inputSample3 + local_z1;
			local_z1 = c_b1 * inputSample3 - c_a1 * outputSample3 + local_z2;
			local_z2 = c_b2 * inputSample3 - c_a2 * outputSample3;
			output[s + 3] = outputSample3;
		}

		// Process remaining samples
		for (; s < sampleFrames; ++s)
		{
			// Biquad filter (Direct Form II Transposed)
			const float inputSample = input[s];
			const float outputSample = c_b0 * inputSample + local_z1;
			local_z1 = c_b1 * inputSample - c_a1 * outputSample + local_z2;
			local_z2 = c_b2 * inputSample - c_a2 * outputSample;
			output[s] = outputSample;
		}

		// Write back state
		z1 = local_z1;
		z2 = local_z2;
	}

	void onSetPins() override
	{
		const bool isStreaming = pinInput.isStreaming();
		pinOutput.setStreaming(isStreaming);

		if (isStreaming)
		{
			updateSampleRateCache();

			// Initialize smoothed values to current pin values
			targetFreq = smoothedFreq = pinFreq.getValue();
			targetGain = smoothedGain = pinGain.getValue();

			setSubProcess(&DBDetroitEQ::subProcess);
		}
		else
		{
			// Reset filter state when not streaming
			z1 = 0.0f;
			z2 = 0.0f;
		}
	}
};

namespace
{
	auto r = Register<DBDetroitEQ>::withId(L"My DB Detroit EQ");
}
