#include "mp_sdk_audio.h"
#include <cmath>
#include <algorithm>

using namespace gmpi;

class DBAutoGainST final : public MpBase2
{
	AudioInPin pinOriginalL;
	AudioInPin pinOriginalR;
	AudioInPin pinEffectL;
	AudioInPin pinEffectR;
	BoolInPin pinOnOff;
	AudioOutPin pinOutputL;
	AudioOutPin pinOutputR;

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
	DBAutoGainST()
	{
		initializePin( pinOriginalL );
		initializePin( pinOriginalR );
		initializePin( pinEffectL );
		initializePin( pinEffectR );
		initializePin( pinOnOff );
		initializePin( pinOutputL );
		initializePin( pinOutputR );
	}

	void subProcess( int sampleFrames )
	{
		auto originalL = getBuffer(pinOriginalL);
		auto originalR = getBuffer(pinOriginalR);
		auto effectL = getBuffer(pinEffectL);
		auto effectR = getBuffer(pinEffectR);
		auto outputL = getBuffer(pinOutputL);
		auto outputR = getBuffer(pinOutputR);

		// Bypass mode: pass effect signal through unchanged
		if (!autoGainEnabled)
		{
			for( int s = sampleFrames; s > 0; --s )
			{
				*outputL++ = *effectL++;
				*outputR++ = *effectR++;
			}
			return;
		}

		// Calculate RMS (Root Mean Square) energy of both signals
		float originalSumSq = 0.0f;
		float effectSumSq = 0.0f;

		const float* origL = originalL;
		const float* origR = originalR;
		const float* effL = effectL;
		const float* effR = effectR;

		for( int s = sampleFrames; s > 0; --s )
		{
			const float oL = *origL++;
			const float oR = *origR++;
			const float eL = *effL++;
			const float eR = *effR++;

			originalSumSq += oL * oL + oR * oR;
			effectSumSq += eL * eL + eR * eR;
		}

		// Compute RMS values (divide by 2 channels)
		const float invSampleCount = 1.0f / (sampleFrames * 2.0f);
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

			*outputL++ = *effectL++ * currentGain;
			*outputR++ = *effectR++ * currentGain;
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

		pinOutputL.setStreaming(true);
		pinOutputR.setStreaming(true);

		setSubProcess(&DBAutoGainST::subProcess);
	}
};

namespace
{
	auto r = Register<DBAutoGainST>::withId(L"My DB Auto Gain ST");
}
