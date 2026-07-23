#pragma once

#include "math/math.hpp"
#include "geometry/geometry.hpp"
#include "utility/utility.hpp"


namespace gutil {
	/////////////////////////////////////////////////////////////////
	/// A small container for points
	/////////////////////////////////////////////////////////////////
	template<IsPoint PointType, int MaxSize>
	struct PointContainer {
		/////////////////////////////////////////////////////////////////
		/// Aliases
		////////////////////////////////////////////////////////////////
		using point_type = PointType;
		using scalar_type = typename PointType::scalar_type;
		static constexpr int DIMENSION = PointType::DIMENSION;
		static constexpr int MAX_SIZE = MaxSize;
		
		////////////////////////////////////////////////////////////////
		/// Data
		////////////////////////////////////////////////////////////////
		int size = 0;
		point_type data[MAX_SIZE];
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr point_type& operator[](int i) noexcept {
			GUTIL_ASSERT( 0<=i && i<size);
			return data[i];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr const point_type& operator[](int i) const noexcept {
			GUTIL_ASSERT( 0<=i && i<size);
			return data[i];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr point_type& operator()(int i) noexcept {
			GUTIL_ASSERT(size>0);
			return data[i%size];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr const point_type& operator()(int i) const noexcept {
			GUTIL_ASSERT(size>0);
			return data[i%size];
		}

		template<typename... Args> requires ( (sizeof...(Args) <= DIMENSION+1) && (std::same_as<point_type,std::decay_t<Args>> &&...))
		void set(Args&&... args) noexcept {
			size = static_cast<int>(sizeof...(Args));
			int i=0;
			( (data[i++] = std::forward<Args>(args)), ...); //increment i during the fold expression
		}

		constexpr void push_back(point_type point) noexcept {
			GUTIL_ASSERT(size<MAX_SIZE);
			data[size++] = std::move(point);
		}

		[[nodiscard]] std::span<point_type> as_span() noexcept { return std::span<point_type>{data, size}; }
		[[nodiscard]] std::span<const point_type> as_span() const noexcept { return std::span<const point_type>{data, size}; }
	};



	/////////////////////////////////////////////////////////////////
	/// Simplex in (usuall in 2D or 3D)
	/////////////////////////////////////////////////////////////////
	template<IsPoint PointType>
	struct Simplex : PointContainer<PointType, PointType::DIMENSION+1> {
		/////////////////////////////////////////////////////////////////
		/// Aliases
		////////////////////////////////////////////////////////////////
		using BASE = PointContainer<PointType,PointType::DIMENSION+1>;
		using BASE::DIMENSION;
		using BASE::data;
		using BASE::size;
		using point_type = typename BASE::point_type;
		using scalar_type = typename BASE::scalar_type;


		////////////////////////////////////////////////////////////////
		/// Simple queries
		////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr scalar_type signed_volume() const noexcept requires(DIMENSION==3) {
			GUTIL_ASSERT( size==4 );
			const point_type A = data[1]-data[0];
			const point_type B = data[2]-data[0];
			const point_type C = data[3]-data[0];

			constexpr scalar_type ONE_SIXTH = scalar_type{1}/scalar_type{6};
			return ONE_SIXTH * ( A[0]*(B[1]*C[2] - C[1]*B[2]) 
							   - B[0]*(A[1]*C[2] - C[1]*A[2]) 
							   + C[0]*(A[1]*B[2] - B[1]*A[2]));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr scalar_type volume() const noexcept requires(DIMENSION==3) {
			return gutil::abs( signed_volume() );
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr scalar_type area(int v) const noexcept requires(DIMENSION==3) {
			GUTIL_ASSERT( size==4 );
			const point_type& A = data[(v+1)%4];
			const point_type& B = data[(v+2)%4];
			const point_type& C = data[(v+3)%4];

			return scalar_type{0.5} * gutil::norm2( gutil::cross(A-C, B-C) );
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr scalar_type height(int v) const noexcept requires(DIMENSION==3) {
			return scalar_type{3} * volume() / area(v);
		}
	};


	/////////////////////////////////////////////////////////////////
	/// Triangle in 3D
	/////////////////////////////////////////////////////////////////
	template<IsScalar T>
	struct Triangle3D : PointContainer<Point<3,T>, 3> {
		/////////////////////////////////////////////////////////////
		/// Aliases
		/////////////////////////////////////////////////////////////
		using BASE = PointContainer<Point<3,T>,3>;
		using point_type = Point<3,T>;
		using scalar_type = T;
		using BASE::DIMENSION;
		using BASE::data;
		using BASE::size;

		
		/////////////////////////////////////////////////////////////
		/// Simple queries
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] point_type normal() const noexcept {
			//counter-clockwise positive orientation
			//  0 - 1				0 - 2
			//   \ /				 \ /
			//    2 				  1
			// oriented down	   oriented up
			return gutil::cross(data[1]-data[0],data[2]-data[0]);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] scalar_type signed_normal_distance(const point_type& p) const noexcept {
			const point_type N = gutil::normalized(normal());
			return gutil::dot( N, p-data[0] );
		}

		[[nodiscard]] scalar_type normal_distance_squared(const point_type& p) const noexcept {
			const point_type N = normal();
			const scalar_type dd = gutil::dot(N,p-data[0]);
			return (dd*dd) / gutil::squared_norm(N);
		}
	};



}