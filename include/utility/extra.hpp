#pragma once

#include <span>
#include <iostream>
#include <string_view>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gutil {
	template<typename T>
	void print_to_stream(std::ostream& os, std::span<const T> data, const std::string_view delimiter) {
		for (const T& val : data) {
			os << val << delimiter;
		}
	}

	template<int DIM, typename T> requires (DIM>0)
	void print_to_stream(std::ostream& os, std::span<const T, DIM> data, const std::string_view delimiter) {
		for (int i=0; i<DIM-1; ++i) {
			os << data[i] << delimiter;
		}
		os << data[DIM-1];
	}


	//Use a Customization Point Object for a generic to_string method
	namespace _cpo_ {
		void to_string() = delete;
		struct to_string_fn final {
			template<typename T>
			[[nodiscard]] GUTIL_STATIC_CALL std::string operator()(const T& val) GUTIL_STATIC_CALL_CONST {
				using DecayT = std::decay_t<T>;
				if constexpr (std::same_as<DecayT,std::string>) {
					return val;
				}
				else if constexpr (std::is_convertible_v<DecayT, std::string>) {
					return std::string{val};
				}
				else if constexpr (requires { to_string(val); }) {
					//custom to_string should end up here
					return to_string(val);
				}
				else if constexpr (requires { std::to_string(val); }) {
					//int, float, etc should end up here
					return std::to_string(val);
				}
				else if constexpr (requires { std::cout << val; }) {
					//if operator<< is defined for custom types
					std::stringstream ss;
					ss << val;
					return ss.str();
				}
				else {
					static_assert(always_false_v<T>, "gutil::to_string - no function found");
					return "";
				}
			}
		};
	}
	inline constexpr _cpo_::to_string_fn to_string{};



	//A log class to print to an output stream in a thread safe manner
	//It also tracks the time elapsed since the beginning of the program and which thread called it
	struct Logger {
		//starting time of the program, sychronization mutex, and output stream
		static inline std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
		static inline std::mutex mtx{};
		static inline std::ostream* out = &std::cout;
		static inline std::ostream* err = &std::cerr;

		//change the output stream
		static void set_output(std::ostream& os) {out = &os;}
		static void set_error(std::ostream& os) {err = &os;}

		//write to the ouput stream (thread safe)
		template<typename... Ts>
		static void log(const Ts&... args) {
			const auto now = std::chrono::steady_clock::now();
			const double elapsed = std::chrono::duration<double>(now - start_time).count();

			std::lock_guard<std::mutex> lock(mtx);

			#ifdef _OPENMP
			*out   << "[t=" << std::fixed << std::setprecision(4) << elapsed << "s | "
					"thread=" << std::this_thread::get_id() << ", " << "omp_thread=" << omp_get_thread_num() << "] "
					<< (gutil::to_string(args) + ...) << "\n";
			#else
			*out   << "[t=" << std::fixed << std::setprecision(4) << elapsed << "s | "
					<< "thread=" << std::this_thread::get_id() << "] "
					<< (gutil::to_string(args) + ...) << "\n";
			#endif

			out -> flush();
		}

		template<typename... Ts>
		static void error(const Ts&... args) {
			const auto now = std::chrono::steady_clock::now();
			const double elapsed = std::chrono::duration<double>(now - start_time).count();

			std::lock_guard<std::mutex> lock(mtx);
			#ifdef _OPENMP
			*err   << "[t=" << std::fixed << std::setprecision(4) << elapsed << "s | "
					"thread=" << std::this_thread::get_id() << ", " << "omp_thread=" << omp_get_thread_num() << "] "
					<< (gutil::to_string(args) + ...) << "\n";
			#else
			*err   << "[t=" << std::fixed << std::setprecision(4) << elapsed << "s | "
					<< "thread=" << std::this_thread::get_id() << "] "
					<< (gutil::to_string(args) + ...) << "\n";
			#endif

			err -> flush();
		}
	};


	//A logging method built to time the duration of some subroutine
	//Initialize it with a label and it prints on 
	struct LogTime {
		const std::string label;
		std::chrono::steady_clock::time_point mark_start;

		explicit LogTime(const std::string label) : label{label}, mark_start{std::chrono::steady_clock::now()} {}

		~LogTime() {
			const auto now = std::chrono::steady_clock::now();
			const double elapsed = std::chrono::duration<double>(now - mark_start).count();
			Logger::log(label + " : " + std::to_string(elapsed) + "s");
		}
	};

}