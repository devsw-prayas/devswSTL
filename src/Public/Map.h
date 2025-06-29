#pragma once
#include "devswSTL.h"
#include <optional>

namespace devsw::stl::abstraction{
    template<typename K, typename V>
    class devswSTL Map{
        using value = V;
        using key = K;

    public:
		Map() = default;
		virtual ~Map() = default;

        //Accessors
        virtual bool insert(const value& valueItem, const key& keyItem) = 0;
        virtual bool insert(const std::pair<key, value> entry) = 0;
        [[nodiscard]] virtual bool contains(const key& keyItem) = 0;
        [[nodiscard]] virtual std::optional<value> get(const key& keyItem) = 0;
        virtual value& operator[](const key& keyItem) = 0;
        [[nodiscard]] virtual std::optional<value> remove(const key& keyItem) = 0;

        //Capacity
        [[nodiscard]] virtual size_t size() const = 0;
        [[nodiscard]] virtual bool isEmpty() const = 0;
    };
}