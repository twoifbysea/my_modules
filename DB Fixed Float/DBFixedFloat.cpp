#include "mp_sdk_audio.h"

using namespace gmpi;

class DBFixedFloat final : public MpBase2
{
	FloatInPin pinValue;
	FloatOutPin pinOut;

public:
	DBFixedFloat()
	{
		initializePin(pinValue);
		initializePin(pinOut);
	}

	void onSetPins() override
	{
		if (pinValue.isUpdated())
		{
			pinOut = pinValue.getValue();
		}
	}
};

namespace
{
	auto r = Register<DBFixedFloat>::withId(L"DB Fixed Float");
}
