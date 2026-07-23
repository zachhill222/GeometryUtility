#pragma once

#include "math/math.hpp"
#include "geometry/geometry.hpp"
#include "utility/utility.hpp"

#include <concepts>

#ifndef GUTIL_MAX_GJK_ITERATIONS
	#define GUTIL_MAX_GJK_ITERATIONS 16
#endif


namespace gutil {
	//SA and SB are two classes that represent convex shapes with a methods:
	//	Point support(const Point& direction)
	//	Point center()

	// FOR DETAILS, SEE:
	// “Implementing GJK - 2006”
	// by Casey Muratori
	// and 
	// https://cs.brown.edu/courses/cs195u/lectures/04_advancedCollisionsAndPhysics.pdf


	/////////////////////////////////////////////////////////////////////
	/// Helper simplex type
	/////////////////////////////////////////////////////////////////////
	template<IsPoint PointType>
	struct SimplexGJK : Simplex<PointType> {
		/////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		////////////////////////////////////////////////////////////////
		using BASE = Simplex<PointType>;
		using BASE::size;
		using BASE::data;
		using BASE::DIMENSION;

		using point_type = typename BASE::point_type;
		using scalar_type = typename BASE::scalar_type;

		static constexpr scalar_type ZERO_TOL = scalar_type{4} * gutil::Epsilon<scalar_type>::value;

		static_assert( IsReal<scalar_type>, "SimplexGJK - scalar type must be real for GJK");


		////////////////////////////////////////////////////////////////
		/// Helper methods for GJK
		////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr bool line_case(point_type& direction) noexcept;
		[[nodiscard]] constexpr bool triangle_case(point_type& direction) noexcept requires(DIMENSION==2);
		[[nodiscard]] constexpr bool triangle_case(point_type& direction) noexcept requires(DIMENSION==3);
		[[nodiscard]] constexpr bool tetra_case(point_type& direction) noexcept requires(DIMENSION==3);
		[[nodiscard]] constexpr bool tetra_case(point_type&) noexcept requires(DIMENSION==2) = delete;
		[[nodiscard]] constexpr bool gjk_iteration(point_type& direction) noexcept;
	};


	template<IsPoint PointType>
	[[nodiscard]] constexpr bool SimplexGJK<PointType>::gjk_iteration(point_type& direction) noexcept {
		//simplex must contain between 2 and 4 points initially
		//simplex and direction will both be updated for the next iteration
		
		//GET NEW SEARCH DIRECTION
		switch (this->size){
			case 2: return line_case(direction);
			case 3: return triangle_case(direction);
			case 4: return tetra_case(direction);
		}

		GUTIL_ABORT("gjk_iteration - unknown simplex size");
		return false;
	}


	template<IsPoint PointType>
	[[nodiscard]] constexpr bool SimplexGJK<PointType>::line_case(point_type& direction) noexcept {
		GUTIL_ASSERT( this->size==2 );

		point_type& A = data[1]; //most recent point
		point_type& B = data[0];

		point_type AO = -A;
		point_type AB = B-A;
		
		if (gutil::dot(AB,AO) > scalar_type{0}) {
			if constexpr (DIMENSION==3) { direction = gutil::cross(AB, gutil::cross(AO,AB)); }
			else if constexpr (DIMENSION==2) {
				point_type AB_perp{-AB[1], AB[0]};
				direction = gutil::dot(AB_perp, AO) < scalar_type{0} ? -AB_perp : AB_perp;
			}

			//check if line segment contained the origin. AB and AO are co-linear.
			if (gutil::squared_norm(direction) <= ZERO_TOL*ZERO_TOL) { return true; }
		}
		else {
			direction = AO;
			this->set(A);
		}
		return false;
	}

	template<IsPoint PointType>
	[[nodiscard]] constexpr bool SimplexGJK<PointType>::triangle_case(point_type& direction) noexcept requires (DIMENSION==2) {
		GUTIL_ASSERT(this->size==3);	//full simplex case for 2D

		point_type& A = data[2]; //most recent point
		point_type& B = data[1];
		point_type& C = data[0];

		point_type AO = -A;
		point_type AB = B-A;
		point_type AC = C-A;

		// 2D perpendiculars (90 degree rotation)
		point_type AB_perp{-AB[1], AB[0]};  // perpendicular to AB
		point_type AC_perp{AC[1], -AC[0]};   // perpendicular to AC
		
		// Make sure perpendiculars point away from triangle
		// If AB_perp points toward C, flip it
		if (gutil::dot(AB_perp, AC) > scalar_type{0}) {
			AB_perp = -AB_perp;
		}
		if (gutil::dot(AC_perp, AB) > scalar_type{0}) {
			AC_perp = -AC_perp;
		}

		// Check region near edge AC
		if (gutil::dot(AC_perp, AO) > scalar_type{0}) {
			// Origin is on the AC side
			if (gutil::dot(AC, AO) > scalar_type{0}) {
				// Origin is in AC region
				point_type AC_perp_to_O{-AC[1], AC[0]};
				if (gutil::dot(AC_perp_to_O, AO) < scalar_type{0}) {
					AC_perp_to_O = -AC_perp_to_O;
				}
				direction = AC_perp_to_O;
				this->set(C,A);
			} else {
				// Origin is in A region
				direction = AO;
				this->set(A);
			}
			return false;
		}

		// Check region near edge AB
		if (gutil::dot(AB_perp, AO) > scalar_type{0}) {
			// Origin is on the AB side
			if (gutil::dot(AB, AO) > scalar_type{0}) {
				// Origin is in AB region
				point_type AB_perp_to_O{-AB[1], AB[0]};
				if (gutil::dot(AB_perp_to_O, AO) < scalar_type{0}) {
					AB_perp_to_O = -AB_perp_to_O;
				}
				direction = AB_perp_to_O;
				this->set(B,A);
			} else {
				// Origin is in A region
				direction = AO;
				this->set(A);
			}
			return false;
		}

		// Origin is inside the triangle
		return true;
	}


	template<IsPoint PointType>
	[[nodiscard]] constexpr bool SimplexGJK<PointType>::triangle_case(point_type& direction) noexcept requires(DIMENSION==3) {
		GUTIL_ASSERT( this->size==3 );

		point_type& A = data[2]; //most recent point
		point_type& B = data[1];
		point_type& C = data[0];

		point_type AO =  -A;
		point_type AB = B-A;
		point_type AC = C-A;

		point_type ABC_normal = gutil::cross(AB,AC); //normal to triangle
		point_type AB_normal  = gutil::cross(AB,ABC_normal); //away from triangle, normal to edge AB, in triangle plane
		point_type AC_normal  = gutil::cross(ABC_normal,AC); //away from triangle, normal to edge AB, in triangle plane

		if (gutil::dot(AC_normal,AO) > scalar_type{0}){
			if (gutil::dot(AC,AO)>scalar_type{0}) {
				direction = gutil::cross(AC, gutil::cross(AO,AC));
				this->set(C,A);
			}
			else {
				if (gutil::dot(AB,AO) > scalar_type{0}) { //STAR
					direction = gutil::cross(AB, gutil::cross(AO,AB));
					this->set(B,A);
				}
				else {
					direction = AO;
					this->set(A);
				}
			}
		}
		else {
			if (gutil::dot(AB_normal,AO)>scalar_type{0}) {
				if (gutil::dot(AB,AO)>scalar_type{0}) { //STAR
					direction = gutil::cross(AB, gutil::cross(AO,AB));
					this->set(B,A);
				}
				else {
					direction = AO;
					this->set(A);
				}
			}
			else {
				//above, below, or on triangle
				const scalar_type dd = gutil::dot(ABC_normal,AO);
				if (dd > ZERO_TOL) {
					direction = ABC_normal;
				}
				else if (dd < -ZERO_TOL) {
					direction = -ABC_normal;
					this->set(B,C,A);
				}
				else {return true;}
			}
		}

		return false;
	}


	template<IsPoint PointType>
	[[nodiscard]] constexpr bool SimplexGJK<PointType>::tetra_case(point_type& direction) noexcept requires(DIMENSION==3) {
		GUTIL_ASSERT(this->size==4);

		point_type& A = data[3]; //most recent point
		point_type& B = data[2];
		point_type& C = data[1];
		point_type& D = data[0];

		constexpr point_type O = point_type::Zeros();

		//get the signed distance to each plane, we know the origin is in the negative side of 
		//the plane BCD because A is the most recent point
		Triangle3D<scalar_type> triangle;
		triangle.set(A,B,C); const scalar_type abc = triangle.signed_normal_distance(O);
		triangle.set(A,D,C); const scalar_type adc = triangle.signed_normal_distance(O);
		triangle.set(A,B,D); const scalar_type abd = triangle.signed_normal_distance(O);
		const scalar_type max_dist = gutil::max(abc, adc, abd);
		if (max_dist < scalar_type{0}) { return true; }

		// reduce to triangle case
		const scalar_type abc_dist = gutil::abs(abc);
		const scalar_type adc_dist = gutil::abs(adc);
		const scalar_type abd_dist = gutil::abs(abd);
		const scalar_type min_dist = gutil::min(abc_dist, adc_dist, abd_dist);

		if (abc_dist == min_dist) { this->set(B,C,A); }
		else if(adc_dist == min_dist) { this->set(C,D,A); }
		else { this->set(D,B,A); }

		// run triangle case
		return triangle_case(direction);
	}

	
	//GJK IMPLEMENTATION
	template<typename SA, typename SB> requires (std::same_as<typename SA::point_type, typename SB::point_type>)
	bool collides(const SA& S1, const SB& S2) {
		using point_type = typename SA::point_type;
		using scalar_type = typename point_type::scalar_type;

		auto minkowski_support = [&S1, &S2](const point_type dir) -> point_type {return S1.support(dir) - S2.support(-dir);};

		//TODO: determine if we can get the center by S.center() or S.center for a better initial condition
		point_type direction = point_type::Zeros();
		direction[0] = scalar_type{1};

		SimplexGJK<point_type> simplex; simplex.set( minkowski_support(direction) );
		direction = -simplex[0];

		//MAIN LOOP
		for (int i=0; i<GUTIL_MAX_GJK_ITERATIONS; ++i) {
			const point_type A = minkowski_support(direction);

			if (gutil::dot(A,direction) < scalar_type{0}){ return false; }

			simplex.push_back(A);

			if (simplex.gjk_iteration(direction)) { return true; }
		}

		return true; //failed to converge, return collision to be safe.
	}


	//use CPO to inject GJK as a fallback method for collision detection 
	// namespace _cpo_ {
	// 	void collides() = delete;	//keep the function name lookup in _cpo_ from seeing gutil::sqrt
	// 	struct collides_fn final {
	// 		template<typename SA, typename SB>
	// 		[[nodiscard]] GUTIL_STATIC_CALL bool operator()(const SA& A, const SB& B) GUTIL_STATIC_CALL_CONST noexcept {
	// 			if constexpr (requires { collides(A,B); }) {
	// 				//check for user defined collides function
	// 				return collides(A,B);
	// 			}
	// 			else if constexpr (requires { collides(B,A); }) {
	// 				//check for user defined collision (symmetric)
	// 				return collides(B,A);
	// 			}
	// 			else if constexpr (requires { gutil::internal::collides(A,B); }) {
	// 				//check for internal implementation
	// 				return gutil::internal::collides(A,B);
	// 			}
	// 			else if constexpr (requires { gutil::internal::collides(B,A); }) {
	// 				//check for internal implementation (symmetric)
	// 				return gutil::internal::collides(B,A);
	// 			}
	// 			else if constexpr (requires { gutil::collides_GJK<SA,SB>(A,B); }) {
	// 				//fallback - call GJK (shapes must be convex)
	// 				return gutil::collides_GJK<SA,SB>(A,B);
	// 			}
	// 			else {
	// 				//throw a compile error
	// 				static_assert(always_false_v<SA>, "gutil::collides - no function found");
	// 			}
	// 		}
	// 	};
	// }

	// inline constexpr _cpo_::collides_fn collides{};
}
