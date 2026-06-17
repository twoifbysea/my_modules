#include "mp_sdk_gui2.h"

using namespace gmpi;

class DBRescalerFloatGUIGui final : public SeGuiInvisibleBase
{
	inline void rescale()
	{
		const float input = pinInput;
		const float inMin = pinInMin;
		const float inMax = pinInMax;
		const float outMin = pinOutMin;
		const float outMax = pinOutMax;

		if (!std::isfinite(input) || !std::isfinite(inMin) || !std::isfinite(inMax) || 
		    !std::isfinite(outMin) || !std::isfinite(outMax))
		{
			pinOutput = outMin;
			return;
		}

		static constexpr float epsilon = 1e-6f;
		if (std::abs(inMax - inMin) < epsilon)
		{
			pinOutput = outMin;
		}
		else
		{
			const float normalized = (input - inMin) / (inMax - inMin);
			pinOutput = outMin + normalized * (outMax - outMin);
		}
	}

 	void onSetInputParameter()
	{
		rescale();
	}

 	void onSetOutput()
	{
		// pinOutput changed
	}

 	FloatGuiPin pinInput;
 	FloatGuiPin pinInMin;
 	FloatGuiPin pinInMax;
 	FloatGuiPin pinOutMin;
 	FloatGuiPin pinOutMax;
 	FloatGuiPin pinOutput;

public:
	DBRescalerFloatGUIGui()
	{
		initializePin( pinInput, static_cast<MpGuiBaseMemberPtr2>(&DBRescalerFloatGUIGui::onSetInputParameter) );
		initializePin( pinInMin, static_cast<MpGuiBaseMemberPtr2>(&DBRescalerFloatGUIGui::onSetInputParameter) );
		initializePin( pinInMax, static_cast<MpGuiBaseMemberPtr2>(&DBRescalerFloatGUIGui::onSetInputParameter) );
		initializePin( pinOutMin, static_cast<MpGuiBaseMemberPtr2>(&DBRescalerFloatGUIGui::onSetInputParameter) );
		initializePin( pinOutMax, static_cast<MpGuiBaseMemberPtr2>(&DBRescalerFloatGUIGui::onSetInputParameter) );
		initializePin( pinOutput, static_cast<MpGuiBaseMemberPtr2>(&DBRescalerFloatGUIGui::onSetOutput) );
	}

};

namespace
{
	auto r = Register<DBRescalerFloatGUIGui>::withId(L"DB Rescaler Float (GUI)");
}
