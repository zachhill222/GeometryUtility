#pragma once

#include "memory/base_allocator.hpp"

#include <span>
#include <type_traits>
#include <cassert>
#include <mutex>

namespace gutil {
	/////////////////////////////////////////////////////////////////
	/// An allocation policy to pass to nodes. This allows us to use
	/// allocators to improve caching when chasing node pointers.
	/////////////////////////////////////////////////////////////////
	template<typename NodeType, IsAllocator Allocator = NewDeleteAllocator> requires( IsAllocatable<Allocator,NodeType> )
	struct NodeAllocationPolicy {
		

		/////////////////////////////////////////////////////////////////
		/// Collect type and other information
		/////////////////////////////////////////////////////////////////
		using allocator_type = Allocator;
		using node_type = NodeType;
		static constexpr bool IS_RAII_SAFE = IsAllocatorRAII<Allocator>;
		static constexpr bool IS_THREAD_SAFE = IsAllocatorThreadSafe<Allocator>;
		static constexpr bool IS_CONTIGUOUS = IsAllocatorContiguous<Allocator>;
		static constexpr int  N_CHILDREN = NodeType::N_CHILDREN;


		/////////////////////////////////////////////////////////////
		/// If the allocator is not thread safe, make it safe in an ad-hoc manner.
		/////////////////////////////////////////////////////////////
		static inline std::mutex mtx;
		[[maybe_unused]] static std::lock_guard<std::mutex> lock_mutex() requires(!IS_THREAD_SAFE) {
			return std::lock_guard<std::mutex>{mtx};
		}
		[[maybe_unused]] static constexpr int lock_mutex() requires(IS_THREAD_SAFE) {
			return 0;
		}

		static inline TypedAllocatorView<NodeType, Allocator> alloc_view{};	//if using new_delete, this is correcly linked by default.
		NodeAllocationPolicy() = default;
		static inline void set_allocator(Allocator& alloc) { alloc_view.set_allocator(alloc); }

		
		/////////////////////////////////////////////////////////////////
		/// Single node construct and destroy
		/////////////////////////////////////////////////////////////////
		template<typename... Args>
		[[nodiscard]] node_type* construct(Args&&... args) {
			assert(alloc_view);
			auto lock = lock_mutex();
			return alloc_view.construct(std::forward<Args>(args)...);
		}

		void destroy(node_type* ptr) noexcept(std::is_nothrow_destructible_v<node_type>) {
			assert(alloc_view);
			auto lock = lock_mutex();
			alloc_view.destroy(ptr);
		}


		/////////////////////////////////////////////////////////////////
		/// Multiple node construct and destroy
		/////////////////////////////////////////////////////////////////
		node_type** construct_children(node_type* parent) requires(!IS_CONTIGUOUS) {
			assert(alloc_view);
			if (!parent) { return nullptr; }
			//fall back to standard new to hold the array of pointers?
			//otherwise we need a policy to allocate pointers.
			//if this is used, it is imperitive that destroy_tree(root) is called.
			node_type** arr = new node_type*[node_type::N_CHILDREN];
			auto lock = lock_mutex();
			for (int i=0; i<N_CHILDREN; ++i) {
				arr[i] = alloc_view.construct(parent, i);	//sets up bounding box and copies allocator pointer
			}
			return arr;
		}

		void destroy_children(node_type** ptr) noexcept(std::is_nothrow_destructible_v<node_type>) requires(!IS_CONTIGUOUS) {
			assert(alloc_view);
			if (!ptr) { return; }
			{
				auto lock = lock_mutex();
				for (int i=0; i<N_CHILDREN; ++i) {
					alloc_view.destroy( ptr[i] );
				}
			}
			delete[] ptr;
		}

		node_type* construct_children(node_type* parent) requires(IS_CONTIGUOUS) {
			assert(alloc_view);
			node_type* children;

			if (!parent) { return nullptr; }
			{
				auto lock = lock_mutex();
				children = alloc_view.allocate_n((size_t) N_CHILDREN);
			}
			for (int i=0; i<N_CHILDREN; ++i) {
				std::construct_at(children+i, parent, i);
			}
			return children;
		}

		void destroy_children(node_type* ptr) noexcept(std::is_nothrow_destructible_v<node_type>) requires(IS_CONTIGUOUS) {
			assert(alloc_view);
			if (!ptr) { return; }
			auto lock = lock_mutex();
			alloc_view.destroy_n(ptr, (size_t) N_CHILDREN);
		}

		/// Recursively destroy an entire subtree.
		/// Note that node->children is either null or points to all N_CHILDREN when IS_CONTIGUOUS is true.
		/// When IS_CONTIGUOUS is false, node->children is either null or ponits to an array of N_CHILDREN,
		/// with the array allocated by new[] (see allocate_children())
		void destroy_tree(node_type* root) noexcept(std::is_nothrow_destructible_v<node_type>) {
			if (!root) { return; }

			//just invoke the destructor of root
			//the node_type destructor is responsible for freeing children
			alloc_view.destroy(root);
		}
	};



	































































	/////////////////////////////////////////////////////////////////
	/// A simple container interface for storing data in nodes.
	/// Use CRTP so container such as a simple array or a subgrid can
	/// both be used.
	/////////////////////////////////////////////////////////////////
	template<typename Derived>
	struct BaseDataContainer {


		/////////////////////////////////////////////////////////////
		/// Collect data from Derived class.
		/////////////////////////////////////////////////////////////
		using allocator_type = typename Derived::allocator_type;
		using value_type = typename Derived::value_type;
		static constexpr size_t MAX_DATA = Derived::MAX_DATA;
		static constexpr bool IS_RAII_SAFE = Derived::IS_RAII_SAFE;			//do we need to explicitly delete the data?
		static constexpr bool IS_THREAD_SAFE = Derived::IS_THREAD_SAFE;		//do we need to worry about synchronization?


		/////////////////////////////////////////////////////////////
		/// If the allocator is not thread safe, make it safe in an ad-hoc manner.
		/////////////////////////////////////////////////////////////
		static inline std::mutex mtx;
		[[maybe_unused]] static std::lock_guard<std::mutex> lock_mutex() requires(!IS_THREAD_SAFE) {
			return std::lock_guard<std::mutex>{mtx};
		}
		[[maybe_unused]] static constexpr int lock_mutex() requires(IS_THREAD_SAFE) {
			return 0;
		}


		/////////////////////////////////////////////////////////////
		/// Ensure that data is cleaned up if this container goes out
		/// of scope.
		/////////////////////////////////////////////////////////////
		~BaseDataContainer() noexcept { clear(); }

		/////////////////////////////////////////////////////////////
		/// CRTP interface
		/////////////////////////////////////////////////////////////
		void clear() noexcept {
			auto lock = lock_mutex();
			static_cast<Derived*>(this) -> clear_impl();
		}

		/// Check if the data already exists
		[[nodiscard]] bool contains(const value_type& value) const noexcept {
			return static_cast<const Derived*>(this) -> contains_impl(value);
		}

		/// Return the pointer to an existing value if the proxy_value (e.g., a copy of a value)
		/// already exists. Return a nullptr if it fails.
		[[nodiscard]] value_type* find(const value_type& proxy_value) const noexcept {
			return static_cast<const Derived*>(this) -> find_impl(proxy_value);
		}

		/// Return the point to the first existing value for wich the predicate returns true.
		/// If no such value exists, return a nullptr.
		template<typename Predicate>
		[[nodiscard]] value_type* find(Predicate&& pred) const noexcept {
			auto lock = lock_mutex();
			return static_cast<const Derived*>(this) -> find_impl(std::forward<Predicate>(pred));
		}

		/// Insert data (a uniqueness and capacity check should be done before this)
		/// Return a pointer to where the data was inserted, but do not require its use.
		value_type* insert(value_type&& value) noexcept {
			auto lock = lock_mutex();
			return static_cast<Derived*>(this) -> insert_impl(std::move(value));
		}

		value_type* insert(value_type value) noexcept {
			auto lock = lock_mutex();
			return insert(std::move(value));	//move a copy
		}


		/// Construct and insert data (a uniqueness and capacity check should be done before this)
		/// Return a pointer to where the data was inserted, but do not require its use.
		template<typename... Args>
		value_type* emplace(Args&&... args) noexcept {
			auto lock = lock_mutex();
			return static_cast<Derived*>(this) -> emplace_impl(std::forward<Args>(args)...);
		}

		/// Check if the container is full
		[[nodiscard]] bool full() const noexcept {
			return static_cast<const Derived*>(this) -> full_impl();
		}

		/// Check if the container is empty
		[[nodiscard]] bool empty() const noexcept {
			return static_cast<const Derived*>(this) -> empty_impl();
		}

		/// Check the current capacity
		[[nodiscard]] size_t capacity() const noexcept {
			return static_cast<const Derived*>(this) -> capacity_impl();
		}

		/// Check the remaining capacity
		[[nodiscard]] size_t remaining_capacity() const noexcept {
			return static_cast<const Derived*>(this) -> remaining_capacity_impl();
		}

		/// Check the current amount of data
		[[nodiscard]] size_t size() const noexcept {
			return static_cast<const Derived*>(this) -> size_impl();
		}

		/// Get a contiguous view into the data.
		[[nodiscard]] std::span<value_type> as_span() noexcept {
			return static_cast<Derived*>(this) -> as_span_impl();
		}

		[[nodiscard]] std::span<const value_type> as_span() const noexcept {
			return static_cast<const Derived*>(this) -> as_span_impl();
		}

		/// Access data by index
		[[nodiscard]] value_type& operator[](size_t idx) noexcept {
			assert(idx<size());
			return static_cast<Derived*>(this) -> get_data_impl(idx);
		}

		[[nodiscard]] const value_type& operator[](size_t idx) const noexcept {
			assert(idx<size());
			return static_cast<Derived*>(this) -> get_data_impl(idx);
		}
	};



	/////////////////////////////////////////////////////////////////
	/// A default array based storage container
	/////////////////////////////////////////////////////////////////
	template<typename ValueType, size_t MaxData = 64, IsAllocatorContiguous Allocator = NewDeleteAllocator> requires(IsAllocatable<Allocator,ValueType>)
	struct DataBlock : public BaseDataContainer<DataBlock<ValueType, MaxData, Allocator>> {
		

		/////////////////////////////////////////////////////////////
		/// Collect essential information
		/////////////////////////////////////////////////////////////
		using BASE = BaseDataContainer<DataBlock<ValueType, MaxData, Allocator>>;

		using allocator_type = Allocator;
		using value_type = ValueType;
		static constexpr size_t MAX_DATA = MaxData;
		static constexpr bool IS_RAII_SAFE = IsAllocatorRAII<allocator_type>;
		static constexpr bool IS_THREAD_SAFE = IsAllocatorThreadSafe<allocator_type>;
		

		/////////////////////////////////////////////////////////////
		/// Set up actual storage
		/////////////////////////////////////////////////////////////
		static inline TypedAllocatorView<value_type, allocator_type> alloc_view{};
		static inline void set_allocator(Allocator& alloc) { alloc_view.set_allocator(alloc); }
		value_type* data_ = nullptr;
		size_t size_ = 0;


		/////////////////////////////////////////////////////////////
		/// Memory management (move but no copy)
		/////////////////////////////////////////////////////////////
		DataBlock() {}

		~DataBlock() noexcept { clear_impl(); }

		DataBlock(const DataBlock&) = delete;
		DataBlock& operator=(const DataBlock&) = delete;

		DataBlock(DataBlock&& other) noexcept {
			size_ = other.size_;
			data_ = other.data_;
			other.data_ = nullptr;
			other.size_ = 0;
		}

		DataBlock& operator=(DataBlock&& other) noexcept {
			if (this != &other) {
				size_ = other.size_;
				data_ = other.data_;
				other.data_ = nullptr;
				other.size_ = 0;
			}
			return *this;
		}


		/////////////////////////////////////////////////////////////
		/// Implement CRTP interface
		/////////////////////////////////////////////////////////////
		void clear_impl() noexcept {
			assert(alloc_view);
			if (data_) {
				//invoke the data destructors if needed
				if constexpr (!std::is_trivially_destructible_v<value_type>) {
					alloc_view.destroy_n(data_, size_);
				}
				//free the memory
				alloc_view.deallocate_n(data_, MAX_DATA);
			}
		}

		/// Check if the data already exists
		[[nodiscard]] bool contains_impl(const value_type& value) const noexcept {
			assert(alloc_view);
			if (data_) {
				for (size_t i=0; i<size_; ++i) {
					if (value == *(data_ + i)) {return true;}
				}
			}
			return false;
		}

		/// Return a pointer to an existing value if the proxy_value (e.g., a copy of a value)
		/// already exists. Return a nullptr if it fails.
		[[nodiscard]] value_type* find_impl(const value_type& proxy_value) const noexcept {
			assert(alloc_view);
			if (data_) {
				for (size_t i=0; i<size_; ++i) {
					if (proxy_value == *(data_ + i)) {return data_ + i;}
				}
			}
			return nullptr;
		}

		/// Return the point to the first existing value for wich the predicate returns true.
		/// If no such value exists, return a nullptr.
		template<typename Predicate>
		[[nodiscard]] value_type* find_impl(Predicate&& pred) const noexcept {
			assert(alloc_view);
			if (data_) {
				for (size_t i=0; i<size_; ++i) {
					if (pred(*(data_+i))) { return data_ + i; }
				}
			}
			return nullptr;
		}

		/// Insert data (a uniqueness and capacity check should be done before this)
		/// Return a pointer to where the data was inserted, but do not require its use.
		value_type* insert_impl(value_type&& value) noexcept {
			assert(alloc_view);
			if (!data_) {
				data_ = alloc_view.allocate_n(MAX_DATA);
			}
			else if (full_impl()) {return nullptr;}

			//the memory is already allocated, use std::construct directly
			value_type* ptr = std::construct_at(data_ + size_, std::move(value));

			++size_;
			return ptr;
		}

		void insert_impl(std::span<value_type> values) noexcept {
			assert(alloc_view);
			if (!data_) {
				data_ = alloc_view.allocate_n(MAX_DATA);
			}
			assert(values.size() < remaining_capacity_impl());

			std::move(std::make_move_iterator(values.begin()), std::make_move_iterator(values.end()), data_+size_);
		}

		/// Construct and insert data (a uniqueness and capacity check should be done before this)
		/// Return a pointer to where the data was inserted, but do not require its use.
		template<typename... Args>
		value_type* emplace_impl(Args&&... args) noexcept {
			assert(alloc_view);
			if (!data_) {
				data_ = alloc_view.allocate_n(MAX_DATA);
			}
			else if (full_impl()) {return nullptr;}

			//the memory is already allocated, use std::construct directly
			value_type* ptr = std::construct_at(data_ + size_, std::forward<Args>(args)...);

			++size_;
			return ptr;
		}

		/// Check if the container is full
		[[nodiscard]] bool full_impl() const noexcept {
			assert(alloc_view);
			return size_ >= MAX_DATA;
		}

		/// Check if the container is empty
		[[nodiscard]] bool empty_impl() const noexcept {
			assert(alloc_view);
			return !data_;
		}

		/// Check the current capacity
		[[nodiscard]] size_t capacity_impl() const noexcept {
			assert(alloc_view);
			return MAX_DATA;
		}

		/// Check the current size
		[[nodiscard]] size_t size() const noexcept {
			assert(alloc_view);
			return size_;
		}

		/// Check the remaining capacity (be careful of integer underflow if capacity was exceeded)
		[[nodiscard]] size_t remaining_capacity_impl() const noexcept {
			assert(alloc_view);
			return size_ < MAX_DATA ? MAX_DATA-size_ : 0;
		}

		/// Get a contiguous view into the data.
		[[nodiscard]] std::span<value_type> as_span_impl() noexcept {
			assert(alloc_view);
			return data_ ? std::span<value_type>{data_, data_+size_} : std::span<value_type>{};
		}

		[[nodiscard]] std::span<const value_type> as_span_impl() const noexcept {
			assert(alloc_view);
			return data_ ? std::span<const value_type>{data_, data_+size_} : std::span<const value_type>{};
		}

		/// Get individual data
		value_type& get_data_impl(size_t idx) noexcept {
			assert(alloc_view);
			assert(idx<size_);
			return data_[idx];
		}

		const value_type& get_data_impl(size_t idx) const noexcept {
			assert(alloc_view);
			assert(idx<size_);
			return data_[idx];
		}
	};




}