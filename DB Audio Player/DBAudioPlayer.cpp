#include "mp_sdk_audio.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace gmpi;

class DBWavePlayer final : public MpBase2
{
	StringInPin pinFileName;
	BoolInPin pinGate;
	BoolInPin pinLoop;
	AudioOutPin pinLeftOut;
	AudioOutPin pinRightOut;

	std::vector<float> leftSamples_;
	std::vector<float> rightSamples_;
	size_t sampleFrameCount_ = 0;
	double sampleFrameCountD_ = 0.0;
	double playhead_ = 0.0;
	double playIncrement_ = 1.0;
	bool playing_ = false;
	bool gateHigh_ = false;
	bool pendingStop_ = false;
	bool outputStreaming_ = false;

	std::vector<float> stagedLeftSamples_;
	std::vector<float> stagedRightSamples_;
	size_t stagedSampleFrameCount_ = 0;
	double stagedSampleFrameCountD_ = 0.0;
	double stagedPlayIncrement_ = 1.0;
	bool stagedFileReady_ = false;

	enum class FileChangeFadeState
	{
		None,
		FadeOutOld,
		FadeInNew
	};

	FileChangeFadeState fileChangeFadeState_ = FileChangeFadeState::None;
	int fileChangeFadeSamplesTotal_ = 0;
	int fileChangeFadeSamplesRemaining_ = 0;

	static constexpr float kFileChangeFadeSeconds = 0.005f;

	static uint16_t readU16(const uint8_t* data)
	{
		return static_cast<uint16_t>(data[0] | (static_cast<uint16_t>(data[1]) << 8));
	}

	static uint32_t readU32(const uint8_t* data)
	{
		return static_cast<uint32_t>(data[0])
			| (static_cast<uint32_t>(data[1]) << 8)
			| (static_cast<uint32_t>(data[2]) << 16)
			| (static_cast<uint32_t>(data[3]) << 24);
	}

	static bool readWholeFile(const std::wstring& filename, std::vector<char>& bytes)
	{
		bytes.clear();

		if (filename.empty())
		{
			return false;
		}

		FILE* file = nullptr;
#ifdef _WIN32
		if (_wfopen_s(&file, filename.c_str(), L"rb") != 0 || !file)
		{
			return false;
		}
#else
		std::vector<char> narrowName(filename.size() * MB_LEN_MAX + 1, '\0');
		const std::size_t converted = std::wcstombs(narrowName.data(), filename.c_str(), narrowName.size());
		if (converted == static_cast<std::size_t>(-1))
		{
			return false;
		}
		file = std::fopen(narrowName.data(), "rb");
		if (!file)
		{
			return false;
		}
#endif

		if (std::fseek(file, 0, SEEK_END) != 0)
		{
			std::fclose(file);
			return false;
		}

		const auto fileSize = std::ftell(file);
		if (fileSize <= 0)
		{
			std::fclose(file);
			return false;
		}

		std::rewind(file);

		bytes.resize(static_cast<size_t>(fileSize));
		const auto bytesRead = std::fread(bytes.data(), 1, bytes.size(), file);
		std::fclose(file);

		return bytesRead == bytes.size();
	}

	static float decodePcmSample(const uint8_t* data, int bitsPerSample)
	{
		switch (bitsPerSample)
		{
		case 8:
		{
			const int sample = static_cast<int>(data[0]) - 128;
			return static_cast<float>(sample / 128.0f);
		}
		case 16:
		{
			const auto sample = static_cast<int16_t>(readU16(data));
			return static_cast<float>(sample / 32768.0f);
		}
		case 24:
		{
			int32_t sample = static_cast<int32_t>(data[0])
				| (static_cast<int32_t>(data[1]) << 8)
				| (static_cast<int32_t>(data[2]) << 16);

			if (sample & 0x00800000)
			{
				sample |= ~0x00FFFFFF;
			}

			return static_cast<float>(sample / 8388608.0f);
		}
		case 32:
		{
			const auto sample = static_cast<int32_t>(readU32(data));
			return static_cast<float>(sample / 2147483648.0f);
		}
		default:
			return 0.0f;
		}
	}

	static float decodeFloat32Sample(const uint8_t* data)
	{
		float sample = 0.0f;
		std::memcpy(&sample, data, sizeof(sample));
		return sample;
	}

	static float lerp(float a, float b, float t)
	{
		return a + (b - a) * t;
	}

	int getFileChangeFadeSamples() const
	{
		return (std::max)(1, static_cast<int>(getSampleRate() * kFileChangeFadeSeconds));
	}

	void clearAudioData()
	{
		leftSamples_.clear();
		rightSamples_.clear();
		sampleFrameCount_ = 0;
		sampleFrameCountD_ = 0.0;
		playhead_ = 0.0;
		playIncrement_ = 1.0;
		playing_ = false;
		pendingStop_ = false;
		fileChangeFadeState_ = FileChangeFadeState::None;
		fileChangeFadeSamplesTotal_ = 0;
		fileChangeFadeSamplesRemaining_ = 0;
	}

	void clearStagedAudioData()
	{
		stagedLeftSamples_.clear();
		stagedRightSamples_.clear();
		stagedSampleFrameCount_ = 0;
		stagedSampleFrameCountD_ = 0.0;
		stagedPlayIncrement_ = 1.0;
		stagedFileReady_ = false;
	}

	void setOutputStreaming(bool isStreaming, int blockPosition = -1)
	{
		if (outputStreaming_ == isStreaming)
		{
			return;
		}

		outputStreaming_ = isStreaming;
		pinLeftOut.setStreaming(isStreaming, blockPosition);
		pinRightOut.setStreaming(isStreaming, blockPosition);
	}

	void selectSubProcess()
	{
		if (playing_)
		{
			setSleep(false);
			setSubProcess(pinLoop ? &DBWavePlayer::subProcessLoop : &DBWavePlayer::subProcessOneShot);
		}
		else
		{
			setSleep(true);
			setSubProcess(&DBWavePlayer::subProcessSilence);
		}
	}

	void applyStopNow()
	{
		TempBlockPositionSetter blockPosition(this, getBlockPosition());
		setOutputStreaming(false, getBlockPosition());
		setSleep(true);
		setSubProcess(&DBWavePlayer::subProcessSilence);
		pendingStop_ = false;
		fileChangeFadeState_ = FileChangeFadeState::None;
		fileChangeFadeSamplesTotal_ = 0;
		fileChangeFadeSamplesRemaining_ = 0;
	}

	void stopPlayback()
	{
		playing_ = false;
		pendingStop_ = false;
		playhead_ = 0.0;
		fileChangeFadeState_ = FileChangeFadeState::None;
		fileChangeFadeSamplesTotal_ = 0;
		fileChangeFadeSamplesRemaining_ = 0;
		clearStagedAudioData();
		setOutputStreaming(false);
		selectSubProcess();
	}

	void startPlayback()
	{
		if (sampleFrameCount_ == 0)
		{
			stopPlayback();
			return;
		}

		playhead_ = 0.0;
		playing_ = true;
		pendingStop_ = false;
		fileChangeFadeState_ = FileChangeFadeState::None;
		fileChangeFadeSamplesTotal_ = 0;
		fileChangeFadeSamplesRemaining_ = 0;
		setOutputStreaming(true);
		selectSubProcess();
	}

	void updateProcessing()
	{
		if (playing_)
		{
			setOutputStreaming(true);
		}
		else
		{
			setOutputStreaming(false);
		}

		selectSubProcess();
	}

	bool loadWaveFileData(
		const std::wstring& filename,
		std::vector<float>& outLeftSamples,
		std::vector<float>& outRightSamples,
		size_t& outSampleFrameCount,
		double& outSampleFrameCountD,
		double& outPlayIncrement)
	{
		std::vector<char> fileBytes;

		if (!readWholeFile(host.resolveFilename_old(filename), fileBytes))
		{
			return false;
		}

		if (fileBytes.size() < 44)
		{
			return false;
		}

		const auto* bytes = reinterpret_cast<const uint8_t*>(fileBytes.data());

		if (std::memcmp(bytes, "RIFF", 4) != 0 || std::memcmp(bytes + 8, "WAVE", 4) != 0)
		{
			return false;
		}

		uint16_t audioFormat = 0;
		uint16_t channelCount = 0;
		uint32_t sourceSampleRate = 0;
		uint16_t bitsPerSample = 0;
		uint16_t blockAlign = 0;
		const uint8_t* audioData = nullptr;
		uint32_t audioDataSize = 0;
		bool foundFmt = false;
		bool foundData = false;

		size_t offset = 12;
		while (offset + 8 <= fileBytes.size())
		{
			const auto* chunk = bytes + offset;
			const uint32_t chunkSize = readU32(chunk + 4);
			offset += 8;

			if (offset + chunkSize > fileBytes.size())
			{
				return false;
			}

			if (std::memcmp(chunk, "fmt ", 4) == 0)
			{
				if (chunkSize < 16)
				{
					return false;
				}

				const auto* fmt = bytes + offset;
				audioFormat = readU16(fmt + 0);
				channelCount = readU16(fmt + 2);
				sourceSampleRate = readU32(fmt + 4);
				blockAlign = readU16(fmt + 12);
				bitsPerSample = readU16(fmt + 14);

				if (audioFormat == 0xFFFE && chunkSize >= 40)
				{
					audioFormat = readU16(fmt + 24);
				}

				foundFmt = true;
			}
			else if (std::memcmp(chunk, "data", 4) == 0)
			{
				audioData = bytes + offset;
				audioDataSize = chunkSize;
				foundData = true;
			}

			offset += chunkSize + (chunkSize & 1u);
		}

		if (!foundFmt || !foundData || !audioData)
		{
			return false;
		}

		if (channelCount < 1 || blockAlign == 0 || bitsPerSample == 0)
		{
			return false;
		}

		const int bytesPerSample = bitsPerSample / 8;
		if (bytesPerSample <= 0 || blockAlign < static_cast<uint16_t>(bytesPerSample * channelCount))
		{
			return false;
		}

		const bool isPcm = (audioFormat == 1);
		const bool isFloat32 = (audioFormat == 3 && bitsPerSample == 32);

		if (!isPcm && !isFloat32)
		{
			return false;
		}

		const size_t frameCount = audioDataSize / blockAlign;
		if (frameCount == 0)
		{
			return false;
		}

		std::vector<float> newLeft(frameCount);
		std::vector<float> newRight(frameCount);

		for (size_t i = 0; i < frameCount; ++i)
		{
			const auto* frame = audioData + (i * blockAlign);

			auto readChannel = [&](int channelIndex) -> float
			{
				const auto* sampleData = frame + (channelIndex * bytesPerSample);

				if (isFloat32)
				{
					return decodeFloat32Sample(sampleData);
				}

				return decodePcmSample(sampleData, bitsPerSample);
			};

			const float left = readChannel(0);
			const float right = (channelCount > 1) ? readChannel(1) : left;

			newLeft[i] = left;
			newRight[i] = right;
		}

		double increment = 1.0;
		const auto hostSampleRate = static_cast<double>(getSampleRate());
		if (sourceSampleRate > 0 && hostSampleRate > 0.0)
		{
			increment = static_cast<double>(sourceSampleRate) / hostSampleRate;
		}

		outLeftSamples = std::move(newLeft);
		outRightSamples = std::move(newRight);
		outSampleFrameCount = frameCount;
		outSampleFrameCountD = static_cast<double>(frameCount);
		outPlayIncrement = increment;

		return true;
	}

	bool loadCurrentFile()
	{
		std::vector<float> newLeft;
		std::vector<float> newRight;
		size_t newFrameCount = 0;
		double newFrameCountD = 0.0;
		double newIncrement = 1.0;

		if (!loadWaveFileData(pinFileName.getValue(), newLeft, newRight, newFrameCount, newFrameCountD, newIncrement))
		{
			clearAudioData();
			return false;
		}

		leftSamples_ = std::move(newLeft);
		rightSamples_ = std::move(newRight);
		sampleFrameCount_ = newFrameCount;
		sampleFrameCountD_ = newFrameCountD;
		playIncrement_ = newIncrement;
		playhead_ = 0.0;
		pendingStop_ = false;

		return true;
	}

	bool stageCurrentFile()
	{
		std::vector<float> newLeft;
		std::vector<float> newRight;
		size_t newFrameCount = 0;
		double newFrameCountD = 0.0;
		double newIncrement = 1.0;

		if (!loadWaveFileData(pinFileName.getValue(), newLeft, newRight, newFrameCount, newFrameCountD, newIncrement))
		{
			clearStagedAudioData();
			return false;
		}

		stagedLeftSamples_ = std::move(newLeft);
		stagedRightSamples_ = std::move(newRight);
		stagedSampleFrameCount_ = newFrameCount;
		stagedSampleFrameCountD_ = newFrameCountD;
		stagedPlayIncrement_ = newIncrement;
		stagedFileReady_ = true;

		return true;
	}

	void beginSmoothFileChange()
	{
		if (!stagedFileReady_)
		{
			return;
		}

		fileChangeFadeState_ = FileChangeFadeState::FadeOutOld;
		fileChangeFadeSamplesTotal_ = getFileChangeFadeSamples();
		fileChangeFadeSamplesRemaining_ = fileChangeFadeSamplesTotal_;
	}

	void activateStagedFile()
	{
		leftSamples_ = std::move(stagedLeftSamples_);
		rightSamples_ = std::move(stagedRightSamples_);
		sampleFrameCount_ = stagedSampleFrameCount_;
		sampleFrameCountD_ = stagedSampleFrameCountD_;
		playIncrement_ = stagedPlayIncrement_;
		playhead_ = 0.0;
		playing_ = (sampleFrameCount_ > 0);
		pendingStop_ = false;

		clearStagedAudioData();

		fileChangeFadeState_ = FileChangeFadeState::FadeInNew;
		fileChangeFadeSamplesTotal_ = getFileChangeFadeSamples();
		fileChangeFadeSamplesRemaining_ = fileChangeFadeSamplesTotal_;
	}

	float getCurrentFileChangeGain() const
	{
		if (fileChangeFadeSamplesRemaining_ <= 0 || fileChangeFadeSamplesTotal_ <= 0)
		{
			return 1.0f;
		}

		switch (fileChangeFadeState_)
		{
		case FileChangeFadeState::FadeOutOld:
			return static_cast<float>(fileChangeFadeSamplesRemaining_) / static_cast<float>(fileChangeFadeSamplesTotal_);

		case FileChangeFadeState::FadeInNew:
			return 1.0f - static_cast<float>(fileChangeFadeSamplesRemaining_) / static_cast<float>(fileChangeFadeSamplesTotal_);

		default:
			return 1.0f;
		}
	}

	void advanceFileChangeFade()
	{
		if (fileChangeFadeState_ == FileChangeFadeState::None || fileChangeFadeSamplesRemaining_ <= 0)
		{
			return;
		}

		--fileChangeFadeSamplesRemaining_;
		if (fileChangeFadeSamplesRemaining_ > 0)
		{
			return;
		}

		if (fileChangeFadeState_ == FileChangeFadeState::FadeOutOld)
		{
			activateStagedFile();
		}
		else
		{
			fileChangeFadeState_ = FileChangeFadeState::None;
			fileChangeFadeSamplesTotal_ = 0;
			fileChangeFadeSamplesRemaining_ = 0;
		}
	}

	void subProcessSilence(int sampleFrames)
	{
		auto* leftOut = getBuffer(pinLeftOut);
		auto* rightOut = getBuffer(pinRightOut);

		std::fill_n(leftOut, sampleFrames, 0.0f);
		std::fill_n(rightOut, sampleFrames, 0.0f);
	}

	void subProcessOneShot(int sampleFrames)
	{
		if (pendingStop_)
		{
			applyStopNow();
			subProcessSilence(sampleFrames);
			return;
		}

		auto* leftOut = getBuffer(pinLeftOut);
		auto* rightOut = getBuffer(pinRightOut);

		for (int s = 0; s < sampleFrames; ++s)
		{
			const float* left = leftSamples_.data();
			const float* right = rightSamples_.data();
			const auto sampleCount = sampleFrameCount_;
			const auto sampleCountD = sampleFrameCountD_;
			const auto increment = playIncrement_;

			if (sampleCount == 0)
			{
				playing_ = false;
				pendingStop_ = true;

				const auto remain = sampleFrames - s;
				std::fill_n(leftOut, remain, 0.0f);
				std::fill_n(rightOut, remain, 0.0f);
				return;
			}

			if (playhead_ >= sampleCountD)
			{
				playing_ = false;
				pendingStop_ = true;

				const auto remain = sampleFrames - s;
				std::fill_n(leftOut, remain, 0.0f);
				std::fill_n(rightOut, remain, 0.0f);
				return;
			}

			const auto index = static_cast<size_t>(playhead_);
			const auto nextIndex = (std::min)(index + 1, sampleCount - 1);
			const float frac = static_cast<float>(playhead_ - static_cast<double>(index));
			const float gain = getCurrentFileChangeGain();

			*leftOut++ = lerp(left[index], left[nextIndex], frac) * gain;
			*rightOut++ = lerp(right[index], right[nextIndex], frac) * gain;

			playhead_ += increment;
			advanceFileChangeFade();
		}

		if (fileChangeFadeState_ != FileChangeFadeState::FadeOutOld && playhead_ >= sampleFrameCountD_)
		{
			playing_ = false;
			pendingStop_ = true;
		}
	}

	void subProcessLoop(int sampleFrames)
	{
		auto* leftOut = getBuffer(pinLeftOut);
		auto* rightOut = getBuffer(pinRightOut); 

		for (int s = 0; s < sampleFrames; ++s)
		{
			const auto sampleCount = sampleFrameCount_;
			const auto sampleCountD = sampleFrameCountD_;
			const auto increment = playIncrement_;
			const float* left = leftSamples_.data();
			const float* right = rightSamples_.data();

			if (playhead_ >= sampleCountD)
			{
				do
				{
					playhead_ -= sampleCountD;
				} while (playhead_ >= sampleCountD);
			}

			const auto index = static_cast<size_t>(playhead_);
			const auto nextIndex = (std::min)(index + 1, sampleCount - 1);
			const float frac = static_cast<float>(playhead_ - static_cast<double>(index));
			const float gain = getCurrentFileChangeGain();

			*leftOut++ = lerp(left[index], left[nextIndex], frac) * gain;
			*rightOut++ = lerp(right[index], right[nextIndex], frac) * gain;

			playhead_ += increment;
			advanceFileChangeFade();
		}
	}

public:
	DBWavePlayer()
	{
		initializePin(pinFileName);
		initializePin(pinGate);
		initializePin(pinLoop);
		initializePin(pinLeftOut);
		initializePin(pinRightOut);
	}

	void onSetPins() override
	{
		if (pinFileName.isUpdated())
		{
			if (playing_)
			{
				if (stageCurrentFile())
				{
					beginSmoothFileChange();
				}
			}
			else
			{
				const bool restart = static_cast<bool>(pinGate);

				if (loadCurrentFile() && restart)
				{
					startPlayback();
				}
				else if (sampleFrameCount_ == 0)
				{
					stopPlayback();
				}
			}
		}

		if (pinGate.isUpdated())
		{
			const bool newGate = static_cast<bool>(pinGate);

			if (newGate && !gateHigh_)
			{
				startPlayback();
			}

			gateHigh_ = newGate;
		}

		if (pinLoop.isUpdated() && playing_)
		{
			selectSubProcess();
		}

		updateProcessing();
	}
};

namespace
{
	auto r = Register<DBWavePlayer>::withId(L"DB Wave Player");
}
