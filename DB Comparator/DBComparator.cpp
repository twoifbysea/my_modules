#include "mp_sdk_audio.h"
#include <cmath>

using namespace gmpi;

class DBComparator final : public MpBase2
{
	enum CompareMode
	{
		Equal = 0,
		NotEqual = 1,
		LessThan = 2,
		GreaterThan = 3,
		LessThanOrEqual = 4,
		GreaterThanOrEqual = 5
	};

	static constexpr float EPSILON = 1e-6f;

	AudioInPin pinCompareValue1;
	AudioInPin pinCompareValue2;
	IntInPin pinCompareMode;
	AudioInPin pinTrueValue;
	AudioInPin pinFalseValue;
	BoolInPin pinUseFalseValue;
	AudioOutPin pinValueOut;

public:
	DBComparator()
	{
		initializePin( pinCompareValue1 );
		initializePin( pinCompareValue2 );
		initializePin( pinCompareMode );
		initializePin( pinTrueValue );
		initializePin( pinFalseValue );
		initializePin( pinUseFalseValue );
		initializePin( pinValueOut );
	}

	void subProcess( int sampleFrames )
	{
		// get pointers to in/output buffers.
		auto compareValue1 = getBuffer(pinCompareValue1);
		auto compareValue2 = getBuffer(pinCompareValue2);
		auto trueValue = getBuffer(pinTrueValue);
		auto falseValue = getBuffer(pinFalseValue);
		auto valueOut = getBuffer(pinValueOut);

		const int compareMode = pinCompareMode;
		const bool useFalseValue = pinUseFalseValue;
		const float defaultFalseValue = useFalseValue ? 0.0f : 0.0f;

		// Hoist comparison mode selection outside the sample loop
		switch(compareMode)
		{
			case Equal:
				for( int s = sampleFrames; s > 0; --s )
				{
					const bool result = (std::abs(*compareValue1 - *compareValue2) < EPSILON);
					*valueOut = result ? *trueValue : (useFalseValue ? *falseValue : 0.0f);
					++compareValue1; ++compareValue2; ++trueValue; ++falseValue; ++valueOut;
				}
				break;
			case NotEqual:
				for( int s = sampleFrames; s > 0; --s )
				{
					const bool result = (std::abs(*compareValue1 - *compareValue2) >= EPSILON);
					*valueOut = result ? *trueValue : (useFalseValue ? *falseValue : 0.0f);
					++compareValue1; ++compareValue2; ++trueValue; ++falseValue; ++valueOut;
				}
				break;
			case LessThan:
				for( int s = sampleFrames; s > 0; --s )
				{
					const bool result = (*compareValue1 < *compareValue2);
					*valueOut = result ? *trueValue : (useFalseValue ? *falseValue : 0.0f);
					++compareValue1; ++compareValue2; ++trueValue; ++falseValue; ++valueOut;
				}
				break;
			case GreaterThan:
				for( int s = sampleFrames; s > 0; --s )
				{
					const bool result = (*compareValue1 > *compareValue2);
					*valueOut = result ? *trueValue : (useFalseValue ? *falseValue : 0.0f);
					++compareValue1; ++compareValue2; ++trueValue; ++falseValue; ++valueOut;
				}
				break;
			case LessThanOrEqual:
				for( int s = sampleFrames; s > 0; --s )
				{
					const bool result = (*compareValue1 <= *compareValue2);
					*valueOut = result ? *trueValue : (useFalseValue ? *falseValue : 0.0f);
					++compareValue1; ++compareValue2; ++trueValue; ++falseValue; ++valueOut;
				}
				break;
			case GreaterThanOrEqual:
				for( int s = sampleFrames; s > 0; --s )
				{
					const bool result = (*compareValue1 >= *compareValue2);
					*valueOut = result ? *trueValue : (useFalseValue ? *falseValue : 0.0f);
					++compareValue1; ++compareValue2; ++trueValue; ++falseValue; ++valueOut;
				}
				break;
		}
	}

	void onSetPins() override
	{
		// Set state of output audio pins.
		pinValueOut.setStreaming(true);

		// Set processing method.
		setSubProcess(&DBComparator::subProcess);
	}
};

namespace
{
	auto r = Register<DBComparator>::withId(L"My DB Comparator");
}
