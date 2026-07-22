#pragma once


#include <type_traits>
#include <memory>
#include <vector>
#include <array>
#include <span>


namespace gutil {


	////////////////////////////////////////////////////////////////////
	/// A fixed array type that is a mix betweed std::array and std::vector.
	/// Use init() will be called when needed.
	/// Use release() to deallocate memory if needed.
	////////////////////////////////////////////////////////////////////
	template<typename T, size_t N, typename Allocator = void>
	struct FixedArray {


		////////////////////////////////////////////////////////////////
		// Allocator aliases
		////////////////////////////////////////////////////////////////
		using allocator_type = std::conditional_t< std::same_as<Allocator,void>, std::allocator<T>, Allocator>;
		using alloc_traits = std::allocator_traits<allocator_type>;


		////////////////////////////////////////////////////////////////
		///	Data that must be stored.
		////////////////////////////////////////////////////////////////
		allocator_type alloc_{};
		T* data_ = nullptr;
		size_t size_ = 0;
		static constexpr size_t capacity_ = N;


		////////////////////////////////////////////////////////////////
		/// Constructor, destructors, move, copy
		////////////////////////////////////////////////////////////////
		FixedArray() noexcept = default;
		FixedArray(allocator_type alloc) : alloc_{alloc} {}

		//no copy
		FixedArray(const FixedArray&) = delete;
		FixedArray& operator=(const FixedArray&) = delete;

		//move
		FixedArray(FixedArray&& other) noexcept : 
			alloc_(other.alloc_),
			data_(other.data_),
			size_(other.size_) {
			other.data_ = nullptr;
			other.size_ = 0;
		}

		FixedArray& operator=(FixedArray&& other) noexcept {
			if (this != &other) {
				release();
				alloc_ = other.alloc_; data_ = other.data_; size_ = other.size_;
				other.data_ = nullptr; other.size_ = 0;
			}
			return *this;
		}

		~FixedArray() { release(); }


		////////////////////////////////////////////////////////////////
		/// Memory management
		////////////////////////////////////////////////////////////////
		void init()  noexcept {
			if (!data_) {
				data_ = alloc_traits::allocate(alloc_, capacity_);
				size_ = 0;
			}
		}

		void release() noexcept {
			if (data_) {
				if constexpr (!std::is_trivially_destructible_v<T>) { 
					for (size_t i=0; i<size_; ++i) { alloc_traits::destroy(alloc_, data_+i); }
				}
					
				alloc_traits::deallocate(alloc_, data_, capacity_);
				data_ = nullptr; size_ = 0;
			}
		}

		void clear() noexcept {
			if (data_) {
				size_ = 0;
			}
		}

		void clear_entry(size_t idx) noexcept {
			assert(idx < size_);
			--size_;
			std::swap(data_[idx], data_[size_]);
		}

		////////////////////////////////////////////////////////////////
		/// Implement stl container concept with a few other utility functions.
		////////////////////////////////////////////////////////////////
		[[nodiscard]] bool full() const noexcept { return size_ >= capacity_; }
		[[nodiscard]] bool empty() const noexcept { return size_ == 0; }
		[[nodiscard]] size_t size() const noexcept { return size_; }
		[[nodiscard]] constexpr size_t capacity() const noexcept { return capacity_; }
		[[nodiscard]] size_t remaining() const noexcept { return capacity_ - size_; }

		void push_back(T value) noexcept {
			assert(!full());
			init();	//no-op if already initialzied
			alloc_traits::construct(alloc_, data_ + size_, std::move(value));
			++size_;
		}

		template<typename... Args>
		void emplace_back(Args&&... args) noexcept {
			assert(!full());
			init();	//no-op if already initialzied
			 alloc_traits::construct(alloc_, data_ + size_, std::forward<Args>(args)...);
			++size_;
		}

		void push_back_range(std::span<T> values) noexcept {
			assert(values.size() <= remaining());
			init();	//no-op if already initialzied
			for (T& v : values) {
				alloc_traits::construct(alloc_, data_+size_, std::move(v));
				++size_;
			}
		}

		template<typename Predicate>
		[[nodiscard]] T* find(Predicate&& pred) const noexcept {
			for (size_t i=0; i<size_; ++i) {
				if ( pred(data_[i]) ) {return data_ + i;}
			}
			return nullptr;
		};


		[[nodiscard]] T& operator[](size_t i) noexcept { assert(i < size_); return data_[i]; }
		[[nodiscard]] const T& operator[](size_t i) const noexcept { assert(i < size_); return data_[i]; }

		[[nodiscard]] std::span<T> as_span() noexcept {return {data_, size_}; }
		[[nodiscard]] std::span<const T> as_span() const noexcept {return {data_, size_}; }

		T* begin() noexcept { return data_; }
		T* end() noexcept { return data_ + size_; }
		const T* begin() const noexcept { return data_; }
		const T* end() const noexcept { return data_ + size_; }
		const T* cbegin() const noexcept { return data_; }
		const T* cend() const noexcept { return data_ + size_; }
	};



}
