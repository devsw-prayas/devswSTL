#pragma once

#include "devswSTL.h"
#include "Queue.h"
#include <chrono>

namespace devsw::stl::abstraction {
	template<typename T, typename A>
	class devswSTL BlockingQueue : protected Queue<T, A> {
		using allocator = A;
		using item = T;
        using SteadyClock = std::chrono::steady_clock;
        using TimePoint = SteadyClock::time_point;
        using Duration = SteadyClock::duration;

    public:
        //Methods that wait until space is available or an item is available to be popped
        virtual bool pushW(const item& item) = 0;
        [[nodiscard]] virtual item& popW() = 0;

        virtual bool tryPush(const item& item, Duration duration) = 0;
        [[nodiscard]] virtual item& tryPop(Duration duration) = 0;

        virtual bool tryPushUntil(const item& item, TimePoint until) = 0;
        [[nodiscard]] virtual item& tryPopUntil(TimePoint until) = 0;
	};
}