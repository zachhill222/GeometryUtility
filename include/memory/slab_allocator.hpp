#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <memory>
#include <type_traits>

#include <mutex>
#include <atomic>

namespace gutil
{
	//A homogeneous slab allocator. Stores a linked list of "slabs" with constant size and number of "slots" to store data.
	template<typename T, size_t BytesPerSlab = 65536>
	struct SlabPool
	{
		//determine the size and number of slots in each slab
		//the slab size should be the OS page size
		static constexpr size_t SLOTS_PER_SLAB = BytesPerSlab / sizeof(T);
		static constexpr size_t BYTES_PER_SLAB = SLOTS_PER_SLAB * sizeof(T);
		static_assert(SLOTS_PER_SLAB >=2, "SlabPool: T is too large for the slab size");
		
		//ensure that multiple threads don't try to allocate simultaneously
		//TODO: make a thread pool version
		mutable std::mutex mtx;

		//single linked list of slab/block storage
		struct Slab
		{
			//primary storage
			alignas(T) std::byte storage[BYTES_PER_SLAB];
			Slab* next = nullptr;

			T* slot(const size_t i) noexcept {
				assert(i < SLOTS_PER_SLAB);
				//make sure the compiler doesn't do any unexpected optimizations
				return std::launder(reinterpret_cast<T*>(storage + i*sizeof(T)));
			}
		};

		//single linked list of free slots
		//these pointers are stored in the first bytes of the correspondng free slot
		struct FreeSlot {FreeSlot* next = nullptr;};
		static_assert(sizeof(T) >= sizeof(FreeSlot), "SlabPool: T is too small to maintain the free list");

		//track the head of the slab and free linked lists
		Slab* _slab_head_ = nullptr;
		FreeSlot* _free_head_ = nullptr;

		//allocate a new T object and return a pointer to it
		[[nodiscard]] T* allocate() {
			std::lock_guard<std::mutex> lock(mtx);	//TODO: make a better lock-free allocator

			if (_free_head_) {
				//we are pointing to a free slot
				FreeSlot* slot = _free_head_;
				_free_head_ = slot->next;
				return reinterpret_cast<T*>(slot);
			}
			else {
				//we need to allocate a new slab
				Slab* slab  = new Slab();
				slab->next  = _slab_head_;
				_slab_head_ = slab;

				//slot 0 will be returned, put the rest of the slots onto the free list
				//use a reverse order so that slot 1 is the next free slot
				for (size_t ii=SLOTS_PER_SLAB-1; ii>=1; --ii) {
					FreeSlot* f = reinterpret_cast<FreeSlot*>(slab->slot(ii));
					f->next     = _free_head_;
					_free_head_ = f;
				}

				//return slot 0
				return slab->slot(0);
			}
		}

		//deallocate an object and add its slot to the free list
		void deallocate(T* p) noexcept {
			std::lock_guard<std::mutex> lock(mtx);	//TODO: make a better lock-free allocator
			
			assert(p != nullptr);
			//get pointer to the start of the slot (as a FreeSlot)
			FreeSlot* f = reinterpret_cast<FreeSlot*>(p);
			f->next     = _free_head_;
			_free_head_ = f;
		}

		//allocate and construct an object at the next free slot
		template<typename... Args>
		[[nodiscard]] T* construct(Args&&... args) {
			T* p = allocate();
			return std::construct_at(p, std::forward<Args>(args)...);	//invoke the object constructor
		}

		//destroy an object and return its slot to the free list
		void destroy(T* p) noexcept(std::is_nothrow_destructible_v<T>) {
			std::destroy_at(p);	//invoke the object destructor
			deallocate(p);
		}

		//release the slabs. does not invoke destructors of objects, but does return the memory blocks to the system.
		void release() noexcept {
			std::lock_guard<std::mutex> lock(mtx);	//TODO: make a better lock-free allocator
			Slab* s = _slab_head_;
			while (s) {
				Slab* next = s->next;
				delete s;
				s = next;
			}

			_slab_head_ = nullptr;
			_free_head_ = nullptr;
		}

		//move the resources of another slab pool to this
		void join(SlabPool&& other) noexcept {
			std::lock_guard<std::mutex> lock(mtx);	//TODO: make a better lock-free allocator
			std::lock_guard<std::mutex> lock2(other.mtx);	//TODO: make a better lock-free allocator

			if (other._slab_head_ == nullptr) {return;} //no resources to take

			//get the tail of the other slab list, point the this slab list to the tail
			Slab* tail = other._slab_head_;
			while (tail->next) {tail = tail->next;}
			tail->next  = _slab_head_;
			_slab_head_ = other._slab_head_;

			//get the tail of the other free list, point the this free list to the tail
			if (other._free_head_) {
				FreeSlot* f = other._free_head_;
				while (f->next) {f = f->next;}
				f->next     = _free_head_;
				_free_head_ = other._free_head_;
			}

			//make sure the other can't release this memory
			other._slab_head_ = nullptr;
			other._free_head_ = nullptr;
		}

		//count the number of slabs
		size_t n_slabs() const {
			std::lock_guard<std::mutex> lock(mtx);	//TODO: make a better lock-free allocator

			size_t n = 0;
			auto head = _slab_head_;
			while (head) {
				++n;
				head = head->next;
			}
			return n;
		}

		//get total memory being reserved
		size_t bytes_reserved() const noexcept {
			return BYTES_PER_SLAB * n_slabs();
		}

		//lifecyle
		SlabPool() {}
		~SlabPool() {release();}

		//no copying
		SlabPool(const SlabPool&) = delete;
		SlabPool& operator=(const SlabPool&) = delete;

		//moving is ok
		SlabPool(SlabPool&& other) noexcept {join(std::move(other));}
		SlabPool& operator=(SlabPool&& other) noexcept {
			//TODO: technically this isn't thread safe.
			//the lock is data could be added between relese and join
			if (this != &other) {
				release();
				join(std::move(other));
			}
			return *this;
		}
	};
}