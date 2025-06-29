#pragma once

#include "devswSTL.h"
#include "Allocators.h"
#include <new>

namespace devsw::stl {
	template<typename T, size_t Size, size_t Alignment>
	class devswSTL BoundedAllocator final : Allocator<T> {
		alignas(Alignment) std::byte buffer[sizeof(T) * Size] = {};
		size_t offset;

	public:
		BoundedAllocator() noexcept : offset(0) {
			static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
		}

		T* allocate(size_t size) override {
			size_t total = sizeof(T) * size;
			size_t aligned = (total + Alignment - 1) & ~(Alignment - 1);
			if (offset + aligned > sizeof(buffer))
			{
				//TODO Errors
			}

			offset += aligned;
			return reinterpret_cast<T*>(buffer + offset - aligned);
		}

		void reset() noexcept {
			offset = 0;
		}
	};

	template<typename T, size_t Alignment = 64, size_t Stripe = 4096>
	class UnboundedAllocator final : Allocator<T> {
		struct MemChunk {
			std::byte* memBlock;
			size_t capacity;
			size_t offset;
			MemChunk* next;
		};

		MemChunk* head;
		MemChunk* current;

	public:
		UnboundedAllocator(size_t defaultChunkSize = 256) noexcept : head(nullptr), current(nullptr) {
			size_t initialBytes = align_up(defaultChunkSize * sizeof(T), Stripe);
			head = current = new MemChunk{
				new std::byte[initialBytes],
				initialBytes,
				0,
				nullptr
			};
		}

		~UnboundedAllocator() noexcept override {
			while (head) {
				MemChunk* next = head->next;
				delete[] head->memBlock;
				delete head;
				head = next;
			}
		}

		T* allocate(size_t count) override {
			size_t totalBytes = count * sizeof(T);
			size_t paddedSize = align_up(totalBytes, Alignment);
			size_t alignedOffset = align_up(current->offset, Alignment);

			// Not enough space → create new chunk
			if (alignedOffset + paddedSize > current->capacity) {
				size_t grown = std::max(paddedSize, static_cast<size_t>(totalCapacity() * GOLDEN_RATIO));
				size_t newCapacity = align_up(grown, Stripe);

				MemChunk* chunk = new MemChunk{
					new std::byte[newCapacity],
					newCapacity,
					0,
					nullptr
				};

				current->next = chunk;
				current = chunk;
				alignedOffset = 0;
			}

			std::byte* ptr = current->memBlock + alignedOffset;
			current->offset = alignedOffset + paddedSize;
			return reinterpret_cast<T*>(ptr);
		}

		void reset() noexcept {
			MemChunk* chunk = head->next;
			while (chunk) {
				MemChunk* next = chunk->next;
				delete[] chunk->memBlock;
				delete chunk;
				chunk = next;
			}
			head->offset = 0;
			head->next = nullptr;
			current = head;
		}

	private:
		size_t totalCapacity() const {
			size_t sum = 0;
			for (MemChunk* chunk = head; chunk; chunk = chunk->next) {
				sum += chunk->capacity;
			}
			return sum;
		}

		static constexpr size_t align_up(size_t value, size_t alignment) {
			return (value + alignment - 1) & ~(alignment - 1);
		}
	};
}