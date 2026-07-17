#pragma once

#include <utility/utility.hpp>

#include <concepts>
#include <type_traits>
#include <new>
#include <atomic>

namespace gutil {
	template<typename T>
	concept IsAllocator = requires(T t) {
		{ T::IS_THREADSAFE }				-> std::convertible_to<bool>;
		{ T::IS_CONTIGUOUSLY_ALLOCATABLE }	-> std::convertible_to<bool>;
	};

	template<typename T>
	concept IsThreadSafeAllocator = IsAllocator<T> && T::IS_THREADSAFE;

	template<typename T>
	concept IsContiguousAllocator = IsAllocator<T> && T::IS_CONTIGUOUSLY_ALLOCATABLE;

	template<typename A, typename T>
	concept IsAllocatable = IsAllocator<A> && requires(A a, T t) {
		{ a.template allocate<T>() }	-> std::same_as<T*>;
		{ a.template deallocate<T>(std::declval<T*>()) };
		{ a.template destroy<T>(std::declval<T*>()) };
		//the constructor is unkown, but we'll assume it is present if the others are.
	} || requires(A a, T t) {
		{ a.allocate() }	-> std::same_as<T*>;
		{ a.deallocate(std::declval<T*>()) };
		{ a.destroy(std::declval<T*>()) };
	};

	template<typename A, typename T>
	concept IsContiguouslyAllocatable = IsContiguousAllocator<A> && requires(A a, T t) {
		{ a.template allocate_n<T>(size_t{0}) }	-> std::same_as<T*>;
		{ a.template deallocate_n<T>(std::declval<T*>(), size_t{0}) };
		{ a.template destroy_n<T>(std::declval<T*>(), size_t{0}) };
		//the constructor is unkown, but we'll assume it is present if the others are.
	} || requires(A a, T t) {
		{ a.allocate_n(size_t{0}) }	-> std::same_as<T*>;
		{ a.deallocate_n(std::declval<T*>(), size_t{0}) };
		{ a.destroy_n(std::declval<T*>(), size_t{0}) };
	};


	////////////////////////////////////////////////////////////////////
	/// A CRTP interface class for memory allocators. A few helper classes
	///	are also defined.
	////////////////////////////////////////////////////////////////////
	template<typename Derived>
	struct BaseAllocator {
		static constexpr bool IS_THREADSAFE = Derived::IS_THREADSAFE;
		static constexpr bool IS_CONTIGUOUSLY_ALLOCATABLE = Derived::IS_CONTIGUOUSLY_ALLOCATABLE;
		GUTIL_DEBUG(std::atomic<size_t> allocated_bytes;)

		///////////////////////////////////////////////////////////////
		/// Single items
		///////////////////////////////////////////////////////////////
		template<typename T>
		[[nodiscard]] T* allocate() {
			GUTIL_DEBUG(allocated_bytes += sizeof(T));
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template allocate_impl<T>(); }) {
				return self-> template allocate_impl<T>();
			}
			else {
				return self->allocate_impl();
			}
		}

		template<typename T>
		void deallocate(T* ptr) noexcept {
			GUTIL_DEBUG(allocated_bytes -= sizeof(T));
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template deallocate_impl<T>(); }) {
				self-> template deallocate_impl<T>(ptr);
			}
			else {
				self->deallocate_impl(ptr);
			}
		}

		template<typename T, typename... Args>
		[[nodiscard]] T* construct(Args&&... args) {
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template construct_impl<T>(); }) {
				return self-> template construct_impl<T>(std::forward<Args>(args)...);
			}
			else {
				return self->construct_impl(std::forward<Args>(args)...);
			}
		}

		template<typename T>
		void destroy(T* ptr) noexcept(std::is_nothrow_destructible_v<T>) {
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template destroy_impl<T>(); }) {
				self-> template destroy_impl<T>(ptr);
			}
			else {
				self->destroy_impl(ptr);
			}
			
		}


		///////////////////////////////////////////////////////////////
		/// Contiguous blocks
		///////////////////////////////////////////////////////////////
		template<typename T>
		[[nodiscard]] T* allocate_n(size_t n) requires(IS_CONTIGUOUSLY_ALLOCATABLE) {
			GUTIL_DEBUG(allocated_bytes += n*sizeof(T));
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template allocate_n_impl<T>(); }) {
				return self-> template allocate_n_impl<T>(n);
			}
			else {
				return self->allocate_n_impl(n);
			}
			
		}

		template<typename T>
		void deallocate_n(T* ptr, size_t n) noexcept requires(IS_CONTIGUOUSLY_ALLOCATABLE) {
			GUTIL_DEBUG(allocated_bytes -= n*sizeof(T));
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template deallocate_n_impl<T>(); }) {
				self-> template deallocate_n_impl<T>(ptr, n);
			}
			else {
				self->deallocate_n_impl(ptr, n);
			}
			
		}

		template<typename T, typename... Args>
		[[nodiscard]] T* construct_n(size_t n, Args&&... args) requires(IS_CONTIGUOUSLY_ALLOCATABLE) {
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template construct_n_impl<T>(); }) {
				return self-> template construct_n_impl<T>(n, std::forward<Args>(args)...);
			}
			else {
				return self->construct_n_impl(n, std::forward<Args>(args)...);
			}
			
		}

		template<typename T>
		void destroy_n(T* ptr, size_t n) noexcept(std::is_nothrow_destructible_v<T>) requires(IS_CONTIGUOUSLY_ALLOCATABLE) {
			Derived* self = static_cast<Derived*>(this);
			if constexpr (requires { self-> template destroy_n_impl<T>(); }) {
				self-> template destroy_n_impl<T>(ptr, n);
			}
			else {
				self->destroy_n_impl(ptr, n);
			}
			
		}


		///////////////////////////////////////////////////////////////
		/// Monitor memory leaks in debug mode
		///////////////////////////////////////////////////////////////
		~BaseAllocator() {
			GUTIL_DEBUG(Logger::log("Allocator end of life with ", allocated_bytes.load(), " bytes still allocated"));
		}
	};







	////////////////////////////////////////////////////////////////////
	/// A wrapper for standard new/delete allocator. Note that we need to
	/// consider custom allignments (e.g., alignas(64))
	////////////////////////////////////////////////////////////////////
	struct NewDeleteAllocator final : public BaseAllocator<NewDeleteAllocator> {
		static constexpr bool IS_THREADSAFE = true;
		static constexpr bool IS_CONTIGUOUSLY_ALLOCATABLE = true;


		///////////////////////////////////////////////////////////////
		/// Single items
		///////////////////////////////////////////////////////////////
		template<typename T>
		[[nodiscard]] T* allocate_impl() {
			if constexpr (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
				return static_cast<T*>(::operator new(sizeof(T), std::align_val_t{alignof(T)}));
			}
			else {
				return static_cast<T*>(::operator new(sizeof(T)));
			}
		}

		template<typename T>
		void deallocate_impl(T* ptr) noexcept {
			GUTIL_DEBUG(allocated_bytes-=sizeof(T));

			if constexpr (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
				::operator delete(ptr, std::align_val_t{alignof(T)});
			}
			else {
				::operator delete(ptr);
			}
		}

		template<typename T, typename... Args>
		[[nodiscard]] T* construct_impl(Args&&... args) {
			T* p = allocate_impl<T>();
			return std::construct_at(p, std::forward<Args>(args)...);
		}

		template<typename T>
		void destroy_impl(T* ptr) noexcept(std::is_nothrow_destructible_v<T>) {
			std::destroy_at(ptr);
			deallocate_impl<T>(ptr);
		}


		///////////////////////////////////////////////////////////////
		/// Contiguous blocks
		///////////////////////////////////////////////////////////////
		template<typename T>
		[[nodiscard]] T* allocate_n_impl(size_t n) {
			if constexpr (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
				return static_cast<T*>(::operator new[](n*sizeof(T), std::align_val_t{alignof(T)}));
			}
			else {
				return static_cast<T*>(::operator new[](n*sizeof(T)));
			}
		}

		template<typename T>
		void deallocate_n_impl(T* ptr, size_t n) noexcept {
			if constexpr (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
				::operator delete[](ptr, std::align_val_t{alignof(T)});
			}
			else {
				::operator delete[](ptr);
			}
		}

		template<typename T, typename... Args>
		[[nodiscard]] T* construct_n_impl(size_t n, Args&&... args) {
			T* p = allocate_n_impl<T>(n);
			size_t i = 0;
			try {
				for (; i<n; ++i) { std::construct_at(p+i, args...); }
			}
			catch (...) {
				//if a constructor threw, cleanup and re-throw
				std::destroy_n(p,i);
				deallocate_n_impl(p,n);
				throw;
			}
			return p;
		}

		template<typename T>
		void destroy_n_impl(T* ptr, size_t n) noexcept(std::is_nothrow_destructible_v<T>) {
			std::destroy_n(ptr, n);
			deallocate_n_impl<T>(ptr, n);
		}
	};


	static_assert(IsThreadSafeAllocator<NewDeleteAllocator>);
	static_assert(IsContiguousAllocator<NewDeleteAllocator>);
	static_assert(IsAllocatable<NewDeleteAllocator,int>);
	static_assert(IsContiguouslyAllocatable<NewDeleteAllocator,int>);


	///////////////////////////////////////////////////////////////
	/// Often we need an allocator for a single type and we don't want to
	/// supply the template every time. Make a view into an existing allocator.
	///////////////////////////////////////////////////////////////
	template<IsAllocator Allocator, typename T>
	struct TypedAllocatorView {
		static constexpr bool IS_THREADSAFE = Allocator::IS_THREADSAFE;
		static constexpr bool IS_CONTIGUOUSLY_ALLOCATABLE = Allocator::IS_CONTIGUOUSLY_ALLOCATABLE;

		Allocator& alloc;
		TypedAllocatorView(Allocator& a) : alloc(a) {}

		[[nodiscard]] T* allocate() { return alloc.template allocate<T>(); }
		void deallocate(T* p) { alloc.template deallocate<T>(p); }	

		template<typename... Args>
		[[nodiscard]] T* construct(Args&&... args) { return alloc.template construct<T>(std::forward<Args>(args)...); }
		void destroy(T* p) noexcept(std::is_nothrow_destructible_v<T>) { alloc.template destroy<T>(p); }

		[[nodiscard]] T* allocate_n(size_t n) requires(Allocator::IS_CONTIGUOUSLY_ALLOCATABLE) { return alloc.template allocate_n<T>(n); }
		void deallocate_n(T* p, size_t n) requires(Allocator::IS_CONTIGUOUSLY_ALLOCATABLE) { alloc.template deallocate_n<T>(p,n); }	

		template<typename... Args>
		[[nodiscard]] T* construct_n(size_t n, Args&&... args) requires(Allocator::IS_CONTIGUOUSLY_ALLOCATABLE) { return alloc.template construct_n<T>(n, args...); }
		void destroy_n(T* p, size_t n) noexcept(std::is_nothrow_destructible_v<T>) requires(Allocator::IS_CONTIGUOUSLY_ALLOCATABLE) { alloc.template destroy_n<T>(p, n); }
	};

	static_assert(IsThreadSafeAllocator<TypedAllocatorView<NewDeleteAllocator,int>>);
	static_assert(IsContiguousAllocator<TypedAllocatorView<NewDeleteAllocator,int>>);
	static_assert(IsAllocatable<TypedAllocatorView<NewDeleteAllocator,int>,int>);
	static_assert(IsContiguouslyAllocatable<TypedAllocatorView<NewDeleteAllocator,int>,int>);
}