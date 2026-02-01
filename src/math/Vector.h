#pragma once

#include <array>
#include <utility>
#include <iostream>
#include <concepts>
#include <cmath>

#ifdef __CUDACC__
#define GGG_CUDA __host__ __device__
#else
#define GGG_CUDA
#endif

namespace ggg
{
    template<typename T, std::size_t N> requires (std::integral<T> || std::floating_point<T>)
    class Vector
    {
    public:
        static_assert(N > 0);
        using value_type = T;
        static constexpr std::size_t dimension = N;

        constexpr Vector() = default;
        GGG_CUDA explicit constexpr Vector(const std::array<T, N>& v) : m_values(v) {}
        template<typename... Args> requires (sizeof...(Args) == N && (std::convertible_to<Args, T> && ...))
        GGG_CUDA explicit constexpr Vector(const Args&... args) : m_values{args...} {}
        constexpr ~Vector() = default;

        [[nodiscard]] GGG_CUDA constexpr T& operator[](std::size_t i) { return m_values[i]; }
        [[nodiscard]] GGG_CUDA constexpr const T& operator[](std::size_t i) const { return m_values[i]; }
        [[nodiscard]] GGG_CUDA constexpr T& x() requires (dimension >= 1) { return m_values[0]; }
        [[nodiscard]] GGG_CUDA constexpr const T& x() const requires (dimension >= 1) { return m_values[0]; }
        [[nodiscard]] GGG_CUDA constexpr T& y() requires (dimension >= 2) { return m_values[1]; }
        [[nodiscard]] GGG_CUDA constexpr const T& y() const requires (dimension >= 2) { return m_values[1]; }
        [[nodiscard]] GGG_CUDA constexpr T& z() requires (dimension >= 3) { return m_values[2]; }
        [[nodiscard]] GGG_CUDA constexpr const T& z() const requires (dimension >= 3) { return m_values[2]; }
        [[nodiscard]] GGG_CUDA constexpr T& w() requires (dimension >= 4) { return m_values[3]; }
        [[nodiscard]] GGG_CUDA constexpr const T& w() const requires (dimension >= 4) { return m_values[3]; }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator+(const Vector& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] + rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator+(const Vector& lhs, const T& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] + rhs)... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator+(const T& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs + rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        GGG_CUDA constexpr Vector& operator+=(const Vector& rhs)
        {
            *this = *this + rhs;
            return *this;
        }

        GGG_CUDA constexpr Vector& operator+=(const T& rhs)
        {
            *this = *this + rhs;
            return *this;
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator-(const Vector& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] - rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator-(const Vector& lhs, const T& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] - rhs)... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator-(const T& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs - rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        GGG_CUDA constexpr Vector& operator-=(const Vector& rhs)
        {
            *this = *this - rhs;
            return *this;
        }

        GGG_CUDA constexpr Vector& operator-=(const T& rhs)
        {
            *this = *this - rhs;
            return *this;
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator*(const Vector& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] * rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator*(const Vector& lhs, const T& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] * rhs)... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator*(const T& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs * rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        GGG_CUDA constexpr Vector& operator*=(const Vector& rhs)
        {
            *this = *this * rhs;
            return *this;
        }

        GGG_CUDA constexpr Vector& operator*=(const T& rhs)
        {
            *this = *this * rhs;
            return *this;
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator/(const Vector& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] / rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator/(const Vector& lhs, const T& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs.m_values[Is] / rhs)... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector operator/(const T& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (lhs / rhs.m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        GGG_CUDA constexpr Vector& operator/=(const Vector& rhs)
        {
            *this = *this / rhs;
            return *this;
        }

        GGG_CUDA constexpr Vector& operator/=(const T& rhs)
        {
            *this = *this / rhs;
            return *this;
        }

        GGG_CUDA constexpr Vector operator-() const
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> Vector
            {
                return Vector{ (-m_values[Is])... };
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr T Dot(const Vector& lhs, const Vector& rhs)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> T
            {
                return ((lhs.m_values[Is] * rhs.m_values[Is]) + ...);
            } (std::make_index_sequence<dimension>());
        }

        [[nodiscard]] T Length() const requires std::floating_point<T>
        {
            return std::sqrt(Dot(*this, *this));
        }

        [[nodiscard]] float Length() const
        {
            return std::sqrt(static_cast<float>(Dot(*this, *this)));
        }

        [[nodiscard]] GGG_CUDA constexpr float InvLength() const
        {
            float lenSq = static_cast<float>(Dot(*this, *this));
            // quake fast inverse sqrt
            long i;
            float x2, y;
            const float threehalfs = 1.5F;
            x2 = lenSq * 0.5F;
            y  = lenSq;
            i  = * ( long * ) &y;
            i  = 0x5f3759df - ( i >> 1 );
            y  = * ( float * ) &i;
            y  = y * ( threehalfs - ( x2 * y * y ) );
            return y;
        }

        [[nodiscard]] GGG_CUDA constexpr Vector Normalized() const requires std::floating_point<T>
        {
            return *this * static_cast<T>(InvLength());
        }

        [[nodiscard]] friend constexpr bool operator==(const Vector& lhs, const Vector& rhs) = default;
        [[nodiscard]] friend constexpr auto operator<=>(const Vector& lhs, const Vector& rhs) = default;

        friend std::ostream& operator<<(std::ostream& os, const Vector& vec)
        {
            [&] <std::size_t... Is> (std::index_sequence<Is...>)
            {
                os << '{';
                ((os << (Is ? "," : "") << vec.m_values[Is]), ...);
                os << '}';
            } (std::make_index_sequence<dimension>());
            return os;
        }

    private:
        std::array<T, N> m_values{};
    };

    using Vec2f = Vector<float, 2>;
    using Vec3f = Vector<float, 3>;
    using Vec4f = Vector<float, 4>;

    using Vec2d = Vector<double, 2>;
    using Vec3d = Vector<double, 3>;
    using Vec4d = Vector<double, 4>;

    using Vec2u = Vector<std::uint32_t, 2>;
    using Vec3u = Vector<std::uint32_t, 3>;
    using Vec4u = Vector<std::uint32_t, 4>;

    using Vec2i = Vector<std::int32_t, 2>;
    using Vec3i = Vector<std::int32_t, 3>;
    using Vec4i = Vector<std::int32_t, 4>;

}

#undef GGG_CUDA