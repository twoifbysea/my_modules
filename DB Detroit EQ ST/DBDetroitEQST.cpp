#include "mp_sdk_audio.h"
#include <cmath>
#include <algorithm>
#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif
//test5
using namespace gmpi;

class DBDetroitEQST final : public MpBase2
{
	AudioInPin pinInputL;
	AudioInPin pinInputR;
	AudioInPin pinFreqL;
	AudioInPin pinFreqR;
	AudioInPin pinGainL;
	AudioInPin pinGainR;
	AudioOutPin pinOutputL;
	AudioOutPin pinOutputR;

	// Biquad filter state variables - separate for L and R channels
	float z1_L = 0.0f;
	float z2_L = 0.0f;
	float z1_R = 0.0f;
	float z2_R = 0.0f;
	
	// Biquad coefficients (normalized) - separate for L and R channels
	float b0_L = 1.0f, b1_L = 0.0f, b2_L = 0.0f;
	float a1_L = 0.0f, a2_L = 0.0f;
	float b0_R = 1.0f, b1_R = 0.0f, b2_R = 0.0f;
	float a1_R = 0.0f, a2_R = 0.0f;
	
	// Parameter smoothing
	float targetFreqL = 0.0f;
	float targetGainL = 0.0f;
	float targetFreqR = 0.0f;
	float targetGainR = 0.0f;
	
	float smoothedFreqL = 0.0f;
	float smoothedGainL = 0.0f;
	float smoothedFreqR = 0.0f;
	float smoothedGainR = 0.0f;
	
	float currentFreqL = -1.0f;
	float currentGainL = -1.0f;
	float currentFreqR = -1.0f;
	float currentGainR = -1.0f;
	
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
	static constexpr float SMOOTHING_TIME_MS = 0.999f; // 10ms smoothing time
	
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
	DBDetroitEQST()
	{
		initializePin(pinInputL);
		initializePin(pinInputR);
		initializePin(pinFreqL);
		initializePin(pinFreqR);
		initializePin(pinGainL);
		initializePin(pinGainR);
		initializePin(pinOutputL);
		initializePin(pinOutputR);
		enableDenormalHandling();
	}

	void subProcess(int sampleFrames)
	{
		auto inputL = getBuffer(pinInputL);
		auto inputR = getBuffer(pinInputR);
		auto freqL = getBuffer(pinFreqL);
		auto freqR = getBuffer(pinFreqR);
		auto gainL = getBuffer(pinGainL);
		auto gainR = getBuffer(pinGainR);
		auto outputL = getBuffer(pinOutputL);
		auto outputR = getBuffer(pinOutputR);

		// Update target values from pin inputs
		targetFreqL = freqL[0];
		targetFreqR = freqR[0];
		targetGainL = gainL[0];
		targetGainR = gainR[0];
		
		// Smooth parameters towards target
		smoothedFreqL += (targetFreqL - smoothedFreqL) * smoothingCoeff;
		smoothedFreqR += (targetFreqR - smoothedFreqR) * smoothingCoeff;
		smoothedGainL += (targetGainL - smoothedGainL) * smoothingCoeff;
		smoothedGainR += (targetGainR - smoothedGainR) * smoothingCoeff;
		
		// Check if smoothed parameters changed enough to recalculate coefficients
		const bool channelsMatch = (std::abs(smoothedFreqL - smoothedFreqR) < EPSILON) && 
		                           (std::abs(smoothedGainL - smoothedGainR) < EPSILON);
		
		const bool paramsChangedL = hasChanged(smoothedFreqL, currentFreqL) || hasChanged(smoothedGainL, currentGainL);
		const bool paramsChangedR = hasChanged(smoothedFreqR, currentFreqR) || hasChanged(smoothedGainR, currentGainR);
		
		if (paramsChangedL)
		{
			currentFreqL = smoothedFreqL;
			currentGainL = smoothedGainL;
			calculateCoefficients(currentFreqL, currentGainL, b0_L, b1_L, b2_L, a1_L, a2_L);
			
			// If channels match, copy L coefficients to R
			if (channelsMatch)
			{
				currentFreqR = smoothedFreqR;
				currentGainR = smoothedGainR;
				b0_R = b0_L;
				b1_R = b1_L;
				b2_R = b2_L;
				a1_R = a1_L;
				a2_R = a2_L;
			}
		}
		
		// Only recalculate R if channels don't match
		if (paramsChangedR && !channelsMatch)
		{
			currentFreqR = smoothedFreqR;
			currentGainR = smoothedGainR;
			calculateCoefficients(currentFreqR, currentGainR, b0_R, b1_R, b2_R, a1_R, a2_R);
		}

		// Cache coefficients in local variables for better register allocation
		const float c_b0_L = b0_L;
		const float c_b1_L = b1_L;
		const float c_b2_L = b2_L;
		const float c_a1_L = a1_L;
		const float c_a2_L = a2_L;
		
		const float c_b0_R = b0_R;
		const float c_b1_R = b1_R;
		const float c_b2_R = b2_R;
		const float c_a1_R = a1_R;
		const float c_a2_R = a2_R;
		
		float local_z1_L = z1_L;
		float local_z2_L = z2_L; 
		float local_z1_R = z1_R;
		float local_z2_R = z2_R;

		// Process both channels with loop unrolling for better performance
		int s = 0;
		const int unrollCount = sampleFrames & ~3;
		
		for(; s < unrollCount; s += 4)
		{
			// Unrolled iteration 0
			const float inputSampleL0 = inputL[s];
			const float outputSampleL0 = c_b0_L * inputSampleL0 + local_z1_L;
			local_z1_L = c_b1_L * inputSampleL0 - c_a1_L * outputSampleL0 + local_z2_L;
			local_z2_L = c_b2_L * inputSampleL0 - c_a2_L * outputSampleL0;
			outputL[s] = outputSampleL0;
			
			const float inputSampleR0 = inputR[s];
			const float outputSampleR0 = c_b0_R * inputSampleR0 + local_z1_R;
			local_z1_R = c_b1_R * inputSampleR0 - c_a1_R * outputSampleR0 + local_z2_R;
			local_z2_R = c_b2_R * inputSampleR0 - c_a2_R * outputSampleR0;
			outputR[s] = outputSampleR0;
			
			// Unrolled iteration 1
			const float inputSampleL1 = inputL[s + 1];
			const float outputSampleL1 = c_b0_L * inputSampleL1 + local_z1_L;
			local_z1_L = c_b1_L * inputSampleL1 - c_a1_L * outputSampleL1 + local_z2_L;
			local_z2_L = c_b2_L * inputSampleL1 - c_a2_L * outputSampleL1;
			outputL[s + 1] = outputSampleL1;
			
			const float inputSampleR1 = inputR[s + 1];
			const float outputSampleR1 = c_b0_R * inputSampleR1 + local_z1_R;
			local_z1_R = c_b1_R * inputSampleR1 - c_a1_R * outputSampleR1 + local_z2_R;
			local_z2_R = c_b2_R * inputSampleR1 - c_a2_R * outputSampleR1;
			outputR[s + 1] = outputSampleR1;
			
			// Unrolled iteration 2
			const float inputSampleL2 = inputL[s + 2];
			const float outputSampleL2 = c_b0_L * inputSampleL2 + local_z1_L;
			local_z1_L = c_b1_L * inputSampleL2 - c_a1_L * outputSampleL2 + local_z2_L;
			local_z2_L = c_b2_L * inputSampleL2 - c_a2_L * outputSampleL2;
			outputL[s + 2] = outputSampleL2;
			
			const float inputSampleR2 = inputR[s + 2];
			const float outputSampleR2 = c_b0_R * inputSampleR2 + local_z1_R;
			local_z1_R = c_b1_R * inputSampleR2 - c_a1_R * outputSampleR2 + local_z2_R;
			local_z2_R = c_b2_R * inputSampleR2 - c_a2_R * outputSampleR2;
			outputR[s + 2] = outputSampleR2;
			
			// Unrolled iteration 3
			const float inputSampleL3 = inputL[s + 3];
			const float outputSampleL3 = c_b0_L * inputSampleL3 + local_z1_L;
			local_z1_L = c_b1_L * inputSampleL3 - c_a1_L * outputSampleL3 + local_z2_L;
			local_z2_L = c_b2_L * inputSampleL3 - c_a2_L * outputSampleL3;
			outputL[s + 3] = outputSampleL3;
			
			const float inputSampleR3 = inputR[s + 3];
			const float outputSampleR3 = c_b0_R * inputSampleR3 + local_z1_R;
			local_z1_R = c_b1_R * inputSampleR3 - c_a1_R * outputSampleR3 + local_z2_R;
			local_z2_R = c_b2_R * inputSampleR3 - c_a2_R * outputSampleR3;
			outputR[s + 3] = outputSampleR3;
		}
		
		// Process remaining samples
		for(; s < sampleFrames; ++s)
		{
			// Left channel - Biquad filter (Direct Form II Transposed)
			const float inputSampleL = inputL[s];
			const float outputSampleL = c_b0_L * inputSampleL + local_z1_L;
			local_z1_L = c_b1_L * inputSampleL - c_a1_L * outputSampleL + local_z2_L;
			local_z2_L = c_b2_L * inputSampleL - c_a2_L * outputSampleL;
			outputL[s] = outputSampleL;
			
			// Right channel - Biquad filter (Direct Form II Transposed)
			const float inputSampleR = inputR[s];
			const float outputSampleR = c_b0_R * inputSampleR + local_z1_R;
			local_z1_R = c_b1_R * inputSampleR - c_a1_R * outputSampleR + local_z2_R;
			local_z2_R = c_b2_R * inputSampleR - c_a2_R * outputSampleR;
			outputR[s] = outputSampleR;
		}
		
		// Write back state
		z1_L = local_z1_L;
		z2_L = local_z2_L;
		z1_R = local_z1_R;
		z2_R = local_z2_R;
	}

	void onSetPins() override
	{
		const bool isStreaming = pinInputL.isStreaming() || pinInputR.isStreaming();
		pinOutputL.setStreaming(isStreaming);
		pinOutputR.setStreaming(isStreaming);

		if (isStreaming)
		{
			updateSampleRateCache();
			
			// Initialize smoothed values to current pin values
			targetFreqL = smoothedFreqL = pinFreqL.getValue();
			targetFreqR = smoothedFreqR = pinFreqR.getValue();
			targetGainL = smoothedGainL = pinGainL.getValue();
			targetGainR = smoothedGainR = pinGainR.getValue();
			
			setSubProcess(&DBDetroitEQST::subProcess);
		}
		else
		{
			// Reset filter state when not streaming
			z1_L = 0.0f;
			z2_L = 0.0f;
			z1_R = 0.0f;
			z2_R = 0.0f;
		}
	}
};

namespace
{
	auto r = Register<DBDetroitEQST>::withId(L"My DB Detroit EQ ST");
}
