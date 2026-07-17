#pragma once

#include "memory/slab_allocator.hpp"

#include <type_traits>
#include <tuple>

namespace gutil {
	//helper types to ensure the types in the allocator are unique
	template<typename T, typename... Ts>
	inline constexpr bool type_in_pack_v = (std::is_same_v<T, Ts> || ...);

	template<typename... Ts>
	struct types_are_unique;

	template<>
	struct types_are_unique<> : std::true_type {};

	template<typename T, typename... Ts>
	struct types_are_unique<T, Ts...> : std::bool_constant<!type_in_pack_v<T,Ts...> && types_are_unique<Ts...>::value> {};

	template<typename... Ts>
	inline constexpr bool types_are_unique_v = types_are_unique<Ts...>::value;

	//an allocator to handle multiple data types. each type gets its own slab allocator.
	template<typename... Ts>
	struct HeteroSlabAllocator {
		static_assert(types_are_unique_v<Ts...>, "HeteroSlabAllocator: data types must be unique");
		static_assert(sizeof...(Ts) > 0, "HeteroSlabAllocator: at least one type must be specified");

		//store one pool per type
		std::tuple<SlabAllocator<Ts>...> _pools_;

		template<typename T> requires (type_in_pack_v<T, Ts...>)
		TypedAllocatorView<SlabAllocator<T>,T> view() {
			return {std::get<SlabAllocator<T>>(_pools_)};
		}

		template<typename T> requires (type_in_pack_v<T, Ts...>)
		SlabAllocator<T>& pool() {
			return std::get<SlabAllocator<T>>(_pools_);
		}

		template<typename T> requires (type_in_pack_v<T, Ts...>)
		const SlabAllocator<T>& pool() const {
			return std::get<SlabAllocator<T>>(_pools_);
		}

		//forward to individual pools
		template<typename T> requires (type_in_pack_v<T, Ts...>)
		[[nodiscard]] T* allocate() {
			return view<T>().allocate();
		}

		template<typename T> requires (type_in_pack_v<T, Ts...>)
		void deallocate(T* p) {
			view<T>().deallocate(p);
		}

		template<typename T, typename... Args> requires (type_in_pack_v<T, Ts...>)
		[[nodiscard]] T* construct(Args&&... args) {
			return view<T>().construct(std::forward<Args>(args)...);
		}

		template<typename T> requires (type_in_pack_v<T, Ts...>)
		void destroy(T* p) noexcept(std::is_nothrow_destructible_v<T>) {
			view<T>().destroy(p);
		}

		//pass to all pools
		void join(HeteroSlabAllocator&& other) {
			(pool<Ts>().join(std::move(other.pool<Ts>())), ...);
		}

		void release() noexcept {
			(pool<Ts>().release(), ...);
		}

		size_t bytes_reserved() const noexcept {
			return (pool<Ts>().bytes_reserved() + ...);
		}

		//lifetime
		HeteroSlabAllocator() = default;
		~HeteroSlabAllocator() = default;

		//no copying
		HeteroSlabAllocator(const HeteroSlabAllocator&) = delete;
		HeteroSlabAllocator& operator=(const HeteroSlabAllocator&) = delete;

		//moving is ok
		HeteroSlabAllocator(HeteroSlabAllocator&&) = default;
		HeteroSlabAllocator& operator=(HeteroSlabAllocator&&) = default;
	};
}