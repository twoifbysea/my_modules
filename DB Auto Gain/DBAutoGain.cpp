#include "mp_sdk_audio.h"
#include <cmath>
#include <algorithm>

using namespace gmpi;

class DBAutoGain final : public MpBase2
{
	AudioInPin pinOriginal;
	AudioInPin pinEffect;
	BoolInPin pinOnOff;
	AudioOutPin pinOutput;

	// Auto gain state
	float currentGain = 1.0f;
	float originalRMSSmoothed = 0.0f;
	float effectRMSSmoothed = 0.0f;
	bool autoGainEnabled = true;
	
	// Smoothing coefficients
	static constexpr float smoothingCoeff = 0.995f;      // Gain smoothing for zipper noise reduction
	static constexpr float attackCoeff = 0.0f;           // Instant attack (0.0 = no smoothing)
	static constexpr float releaseCoeff = 0.9995f;       // Slow release for envelope following
	
	// Safety limits
	static constexpr float minRMS = 0.0001f;             // -80 dB threshold
	
	// Perceptual loudness model
	static constexpr float loudnessExponent = 0.65f;      // Stevens' power law exponent

public:
	DBAutoGain()
	{
		initializePin( pinOriginal );
		initializePin( pinEffect );
		initializePin( pinOnOff );
		initializePin( pinOutput );
	}

	void subProcess( int sampleFrames )
	{
		auto original = getBuffer(pinOriginal);
		auto effect = getBuffer(pinEffect);
		auto output = getBuffer(pinOutput);

		// Bypass mode: pass effect signal through unchanged
		if (!autoGainEnabled)
		{
			for( int s = sampleFrames; s > 0; --s )
			{
				*output++ = *effect++;
			}
			return;
		}

		// Calculate RMS (Root Mean Square) energy of both signals
		float originalSumSq = 0.0f;
		float effectSumSq = 0.0f;

		const float* orig = original;
		const float* eff = effect;

		for( int s = sampleFrames; s > 0; --s )
		{
			const float o = *orig++;
			const float e = *eff++;

			originalSumSq += o * o;
			effectSumSq += e * e;
		}

		// Compute RMS values
		const float invSampleCount = 1.0f / sampleFrames;
		const float originalRMS = std::sqrtf(originalSumSq * invSampleCount);
		const float effectRMS = std::sqrtf(effectSumSq * invSampleCount);

		// Apply envelope smoothing with attack/release characteristics
		const float originalEnvCoeff = (originalRMS > originalRMSSmoothed) ? attackCoeff : releaseCoeff;
		const float effectEnvCoeff = (effectRMS > effectRMSSmoothed) ? attackCoeff : releaseCoeff;
		
		originalRMSSmoothed = originalEnvCoeff * originalRMSSmoothed + (1.0f - originalEnvCoeff) * originalRMS;
		effectRMSSmoothed = effectEnvCoeff * effectRMSSmoothed + (1.0f - effectEnvCoeff) * effectRMS;

		// Calculate perceived loudness using Stevens' power law (loudness ∝ intensity^0.6)
		// This models the non-linear relationship between physical intensity and perceived loudness
		const float originalLoudness = powf((std::max)(originalRMSSmoothed, minRMS), loudnessExponent);
		const float effectLoudness = powf((std::max)(effectRMSSmoothed, minRMS), loudnessExponent);

		// Calculate target gain to match original loudness (no limits)
		float targetGain = 1.0f;
		if (effectLoudness > minRMS)
		{
			targetGain = originalLoudness / effectLoudness;
		}

		// Apply gain with per-sample smoothing to prevent zipper noise
		const float oneMinusSmoothing = 1.0f - smoothingCoeff;
		for( int s = sampleFrames; s > 0; --s )
		{
			currentGain = smoothingCoeff * currentGain + oneMinusSmoothing * targetGain;

			*output++ = *effect++ * currentGain;
		}
	}

	void onSetPins() override
	{
		if( pinOnOff.isUpdated() )
		{
			autoGainEnabled = pinOnOff.getValue();
			
			// Reset state when disabling auto-gain
			if (!autoGainEnabled)
			{
				currentGain = 1.0f;
				originalRMSSmoothed = 0.0f;
				effectRMSSmoothed = 0.0f;
			}
		}

		pinOutput.setStreaming(true);

		setSubProcess(&DBAutoGain::subProcess);
	}
};

namespace
{
	auto r = Register<DBAutoGain>::withId(L"My DB Auto Gain");
}
