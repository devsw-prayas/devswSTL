#pragma once

#include "devswSTL.h"
#include "Iterators.h"
#include "Allocators.h"
#include "List.h"

namespace devsw::stl::implementation {
	template<typename T, typename A = StackAllocator<T>>
	class Vector : abstraction::List<T, A>, Iterable<Vector<T>, T, std::random_access_iterator_tag> {
	};
}