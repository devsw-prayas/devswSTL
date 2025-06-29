#pragma once

#include "devswSTL.h"
#include <type_traits>

namespace devsw::stl::abstraction{
    template<typename T>
    class devswSTL Set{
        using item = T;
    public:

        //Accessors
        virtual bool insert(const item& element) = 0;
        virtual bool remove(const item& element) = 0;
        [[nodiscard]] virtual bool contains(const item& element) const = 0;
        
        //Capacity
        [[nodiscard]] virtual size_t size() const = 0;
        [[nodiscard]] virtual bool isEmpty() const = 0;
        virtual void clear() = 0;

        template<typename ...Args>
        bool emplace(Args&& ...args){
            return insert(item(std::forward<Args>(args)...));
        }

    };
}
