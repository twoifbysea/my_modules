#include "mp_sdk_audio.h"
#include <cmath>
#include <vector>

using namespace gmpi;

class DBTapeMachine final : public MpBase2
{
	AudioInPin pinInputL;
	AudioInPin pinInputR;
	AudioInPin pinDrive;
	AudioInPin pinBias;
	AudioInPin pinWow;
	AudioInPin pinFlutter;
	AudioInPin pinOutput;
	BoolInPin pinOversampling;
	BoolInPin pinOnOff;
	AudioOutPin pinOutputL;
	AudioOutPin pinOutputR;

	// State variables
	std::vector<float> delayBufferL;
	std::vector<float> delayBufferR;
	int delayWritePos;
	float wowPhase;
	float flutterPhase;
	float sampleRate;
	
	// Filter states for tape character (lowpass)
	float filterStateL1, filterStateL2;
	float filterStateR1, filterStateR2;

	// Oversampling filter states
	float prevInputL, prevInputR;
	float prevOutputL, prevOutputR;

public:
	DBTapeMachine()
		: delayWritePos(0)
		, wowPhase(0.0f)
		, flutterPhase(0.0f)
		, sampleRate(44100.0f)
		, filterStateL1(0.0f), filterStateL2(0.0f)
		, filterStateR1(0.0f), filterStateR2(0.0f)
		, prevInputL(0.0f), prevInputR(0.0f)
		, prevOutputL(0.0f), prevOutputR(0.0f)
	{
		initializePin( pinInputL );
		initializePin( pinInputR );
		initializePin( pinDrive );
		initializePin( pinBias );
		initializePin( pinWow );
		initializePin( pinFlutter );
		initializePin( pinOutput );
		initializePin( pinOversampling );
		initializePin( pinOnOff );
		initializePin( pinOutputL );
		initializePin( pinOutputR );
	}

	int32_t open() override
	{ 
		sampleRate = getSampleRate();
		int delayBufferSize = static_cast<int>(sampleRate * 0.5f);
		delayBufferL.resize(delayBufferSize, 0.0f);
		delayBufferR.resize(delayBufferSize, 0.0f);
		return MpBase2::open();
	}

	float tapeClip(float input, float drive, float bias)
	{
		float x = input * (1.0f + drive * 3.0f) + bias * 0.5f;
		return std::tanh(x);
	}

	float lowpass(float input, float& state1, float& state2)
	{
		const float cutoff = 0.75f;
		state1 = state1 + cutoff * (input - state1);
		state2 = state2 + cutoff * (state1 - state2);
		return state2;
	}

	float processSample(float inputL, float inputR, float driveVal, float biasVal, 
						float wowVal, float flutterVal, float gain, 
						float& outL, float& outR, float effectiveSampleRate)
	{
		wowPhase += 0.5f / effectiveSampleRate;
		if (wowPhase >= 1.0f) wowPhase -= 1.0f;
		
		flutterPhase += 8.0f / effectiveSampleRate;
		if (flutterPhase >= 1.0f) flutterPhase -= 1.0f;
		
		float wowMod = std::sin(wowPhase * 2.0f * 3.14159265f) * wowVal;
		float flutterMod = std::sin(flutterPhase * 2.0f * 3.14159265f) * flutterVal;
		float totalMod = (wowMod + flutterMod) * effectiveSampleRate * 0.01f;
		
		// Apply saturation
		float processedL = tapeClip(inputL, driveVal, biasVal);
		float processedR = tapeClip(inputR, driveVal, biasVal);
		
		int delayBufferSize = static_cast<int>(delayBufferL.size());
		
		// Write to delay buffer
		delayBufferL[delayWritePos] = processedL;
		delayBufferR[delayWritePos] = processedR;
		
		// Calculate read position with modulation
		float readPos = delayWritePos - 10.0f - totalMod;
		while (readPos < 0) readPos += delayBufferSize;
		while (readPos >= delayBufferSize) readPos -= delayBufferSize;
		
		// Linear interpolation
		int readPosInt = static_cast<int>(readPos);
		float frac = readPos - readPosInt;
		int readPosNext = (readPosInt + 1) % delayBufferSize;
		
		float delayedL = delayBufferL[readPosInt] * (1.0f - frac) + delayBufferL[readPosNext] * frac;
		float delayedR = delayBufferR[readPosInt] * (1.0f - frac) + delayBufferR[readPosNext] * frac;
		
		// Apply tape character filter and output gain
		outL = lowpass(delayedL, filterStateL1, filterStateL2) * gain;
		outR = lowpass(delayedR, filterStateR1, filterStateR2) * gain;
		
		// Advance write position
		delayWritePos = (delayWritePos + 1) % delayBufferSize;
		
		return 0.0f;
	}

	void subProcess( int sampleFrames )
	{
		// get pointers to in/output buffers.
		auto inputL = getBuffer(pinInputL);
		auto inputR = getBuffer(pinInputR);
		auto drive = getBuffer(pinDrive);
		auto bias = getBuffer(pinBias);
		auto wow = getBuffer(pinWow);
		auto flutter = getBuffer(pinFlutter);
		auto output = getBuffer(pinOutput);
		auto outputL = getBuffer(pinOutputL);
		auto outputR = getBuffer(pinOutputR);

		bool onOff = pinOnOff.getValue();
		bool oversample = pinOversampling.getValue();

		for( int s = sampleFrames; s > 0; --s )
		{
			if (!onOff)
			{
				*outputL = *inputL;
				*outputR = *inputR;
			}
			else
			{
				float gain = std::pow(10.0f, *output / 20.0f);
				float driveVal = *drive;
				float biasVal = *bias;
				float wowVal = *wow * 0.005f;
				float flutterVal = *flutter * 0.002f;
				
				if (oversample)
				{
					// 2x oversampling
					float outL1, outR1, outL2, outR2;
					
					// Upsample: interpolate between previous and current sample
					float interpL = (prevInputL + *inputL) * 0.5f;
					float interpR = (prevInputR + *inputR) * 0.5f;
					
					// Process first oversampled sample (interpolated)
					processSample(interpL, interpR, driveVal, biasVal, wowVal, flutterVal, 
								  gain, outL1, outR1, sampleRate * 2.0f);
					
					// Process second oversampled sample (actual input)
					processSample(*inputL, *inputR, driveVal, biasVal, wowVal, flutterVal, 
								  gain, outL2, outR2, sampleRate * 2.0f);
					
					// Downsample: simple averaging with anti-aliasing
					*outputL = (outL1 + outL2) * 0.5f;
					*outputR = (outR1 + outR2) * 0.5f;
					
					prevInputL = *inputL;
					prevInputR = *inputR;
				}
				else
				{
					// Normal processing (no oversampling)
					processSample(*inputL, *inputR, driveVal, biasVal, wowVal, flutterVal, 
								  gain, *outputL, *outputR, sampleRate);
				}
			}

			// Increment buffer pointers.
			++inputL;
			++inputR;
			++drive;
			++bias;
			++wow;
			++flutter;
			++output;
			++outputL;
			++outputR;
		}
	}

	void onSetPins() override
	{
		// Set state of output audio pins.
		bool streaming = pinInputL.isStreaming() || pinInputR.isStreaming();
		pinOutputL.setStreaming(streaming);
		pinOutputR.setStreaming(streaming);

		// Set processing method.
		setSubProcess(&DBTapeMachine::subProcess);
	}
};

namespace
{
	auto r = Register<DBTapeMachine>::withId(L"My DB Tape Machine");
}
