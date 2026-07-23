#pragma once

#include "utility/macros.hpp"
#include "utility/concepts.hpp"

#include "geometry/point.hpp"

#include <random>
#include <type_traits>

namespace gutil
{
	///////////////////////////////////////////////
	/// For testing, it is helpful to be able to generate
	/// random values. This allows us to generate random
	/// points for standard c++ scalar types.
	///////////////////////////////////////////////
	template<IsPoint PointType, bool Deterministic, typename DistributionType>
	struct RandomPoint {
	private:
		using scalar_type = typename PointType::scalar_type;

		template<size_t N>
		std::array<size_t,N> make_seeds(size_t s1) {
			std::array<size_t,N> arr;
			for (size_t i=0; i<arr.size(); ++i) {
				arr[i] = (Deterministic) ? s1 + i : s1 ^ static_cast<size_t>(rng());
			}
			return arr;
		}

		//define the seed and rng generator
		std::mt19937 gen{};
		DistributionType dist{scalar_type{0}, scalar_type{1}};
		std::random_device rng{};
		std::seed_seq seeds;

	public:
		RandomPoint() {set_seed();}

		//adjust the seed and distribution
		void set_seed(size_t s1 = 0x1234ABCDULL) {
			auto data = make_seeds<16>(s1);
			std::seed_seq seeds{data.begin(), data.end()};
			gen.seed(seeds);
		}

		template<typename... Ts> requires (std::same_as<scalar_type,Ts> && ...)
		void set_parameters(Ts... args) {
			dist = DistributionType{args...};
		}

		//get a random point
		PointType operator()() {
			PointType result;
			for (int i=0; i<PointType::DIMENSION; ++i) {
				result.data[i] = dist(gen);
			}
			return result;
		}

		//get a random scalar
		auto scalar() {
			return dist(gen);
		}
	};


	//factory for making generators
	template<IsPoint PointType, bool Deterministic=true>
	static inline auto UniformRandomPoint() {
		using T = typename PointType::scalar_type;
		if constexpr (IsInteger<T>) {
			return RandomPoint<PointType, Deterministic, std::uniform_int_distribution<T>>{};
		}
		else if constexpr (IsReal<T>) {
			return RandomPoint<PointType, Deterministic, std::uniform_real_distribution<T>>{};
		}
		else {
			static_assert(always_false_v<T>, "UniformRandomPoint - invalid scalar type");
			return 1;
		}
	}

	template<IsPoint PointType, bool Deterministic=true>
	static inline auto NormalRandomPoint() {
		using T = typename PointType::scalar_type;
		if constexpr (IsReal<T>) {
			return RandomPoint<PointType, Deterministic, std::normal_distribution<T>>{};
		}
		else {
			static_assert(always_false_v<T>, "NormalRandomPoint - invalid scalar type");
			return 1;
		}
	}
}




